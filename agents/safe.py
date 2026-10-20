"""Safe PPO agent for photobioreactor (phycocyanin) control.

A Proximal Policy Optimisation agent augmented with a layered safety stack
that keeps the reactor inside its constraints during both training rollouts
and deployment inference.

Pipeline overview::

    Actor (GRU + raw-state skip, agents/networks.py)  ->  tanh squash
         ->  L0 SERL stage mask  ->  L1/L2 APN projection (safety/)
         ->  final SERL mask override  ->  environment
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import copy

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

# ─── Local ────────────────────────────────────────────────────────────────────
from agents.memory import RunningMeanStd
from agents.networks import GRU_HIDDEN_DIM, GRU_NUM_LAYERS, SEQ_LEN, ActorCritic, device
from envs.plant import DEFAULT_SQUASHED, PhycocyaninEnvCore
from safety.apn import load_apn, projection_active
from safety.constants import APN_PATH, SAFE_LOGIT

# ─── Safety Gradient ──────────────────────────────────────────────────────────

# Weight of the APN safety loss (hinge on every head's margin at the actor's
# own mean action) in the PPO loss.  Its gradient is the actor's direct signal
# that an intent would have been corrected.  The APN's margins are logits
# (~30x the physical margins), hence a smaller weight than Fermentation's 1.0.
SAFETY_GRAD_COEF = 0.1

# Weight of the mapping penalty (actor mean -> the projected action it led to).
MAPPING_COEF = 0.01

# ─── PPO Update ───────────────────────────────────────────────────────────────

# TBPTT chunks per gradient step.  One reactor gives one SEQ_LEN row per chunk,
# so chunks are batched: 8 x 32 = 256 samples per step.
CHUNKS_PER_BATCH = 8

# ─── PPO Agent ────────────────────────────────────────────────────────────────


class PPO_Agent:
    """Safe Recurrent PPO agent for photobioreactor control.

    A recurrent actor-critic (GRU + raw-state skip) proposes the action; a
    layered safety stack around it keeps the reactor inside its constraints:

    L0  **SERL stage mask** — locks the action dims a stage does not allow
        (e.g. no outflow outside Harvesting) to their defaults, in the policy
        distribution and again on the action sent.
    L1  **APN** (safety/apn.py) — one learned margin logit per constraint
        (G1 path nitrate, G2 Cq/Cx ratio, G3 harvest-start nitrate still
        reachable, G4 overflow), all from one forward pass.
    L2  **Projection** — sign-gradient ascent on the smooth minimum of the
        margins until the action is accepted (or the best iterate).

    During the PPO update the same margins give the actor a **safety
    gradient**: the APN's smooth hinge on the actor's own (squashed) mean,
    backpropagated through the frozen APN, plus a mapping penalty pulling the
    mean towards the projected action it led to.  PPO itself is trained on the
    raw intent the actor sampled, so its ratios stay on-policy.  The
    environment's Lagrange multipliers shape the reward on top, and for G5
    (end in Idle), which the APN does not cover.

    Training uses Deep Recurrent Policy Optimisation (DRPO): GAE with recurrent
    value estimates and Truncated Backpropagation Through Time (TBPTT) over
    SEQ_LEN-step chunks.  Rewards are normalised online with a Welford running
    estimator to suppress variance from Lagrangian constraint-penalty spikes.

    Attributes:
        gamma: Discount factor for future rewards.
        eps_clip: PPO surrogate clipping range.
        K_epochs: Number of PPO update epochs per learn() call.
        entropy_coeff: Coefficient on the entropy bonus.
        policy: Current actor-critic network.
        policy_old: Old (frozen) actor-critic for rollout collection.
        apn: The frozen APN, or None without a safety layer.
        reward_rms: Running reward normalisation statistics.
        h: Live recurrent hidden state across select_action() calls.
    """

    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma,
                 K_epochs, eps_clip, entropy_coeff, use_safeguard=True,
                 use_projection=True, use_safety_grad=True, use_g3=True):
        """Initialise the safe PPO agent.

        Args:
            state_dim: Observation width (12).
            action_dim: Number of action dimensions (4).
            lr_actor: Initial learning rate for the GRU encoder and actor head.
            lr_critic: Initial learning rate for the critic head.
            gamma: Discount factor for future rewards.
            K_epochs: Number of PPO update epochs per learn() call.
            eps_clip: PPO surrogate clipping range (ratio in [1-eps, 1+eps]).
                Also used as the clipping range for the value loss.
            entropy_coeff: Coefficient on the entropy bonus; annealed by the
                training loop.
            use_safeguard: If True, applies the SERL stage mask and loads the APN.
            use_projection: L2: project every action with the APN.
            use_safety_grad: The APN safety loss and mapping penalty in the PPO
                update.
            use_g3: The APN's G3 head in the projection and the safety gradient.
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coeff = entropy_coeff
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_safeguard = use_safeguard
        self.use_projection = use_projection
        self.use_safety_grad = use_safety_grad

        # 1. Actor-critic (predicts the raw intent z).
        self.policy = ActorCritic(state_dim, action_dim, use_safeguard).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.gru.parameters(),    'lr': lr_actor},
            {'params': self.policy.actor.parameters(),  'lr': lr_actor},
            {'params': [self.policy.log_std],           'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic, 'weight_decay': 1e-5}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, use_safeguard).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 2. The APN: frozen, used for projection and the safety gradient.
        self.apn = None
        if use_safeguard and (use_projection or use_safety_grad):
            self.apn = load_apn(device, state_dim=state_dim, action_dim=action_dim)
            if self.apn is None:
                print(f"WARNING: {APN_PATH} not found (run python -m safety.train); "
                      "actions are not projected and there is no safety gradient.")
            else:
                self.apn.use_g3 = use_g3
        # Projection runs one state at a time: a stream of tiny kernels that a
        # CPU copy executes faster than the GPU.
        self.apn_infer = (copy.deepcopy(self.apn).cpu()
                          if self.apn is not None and device.type != 'cpu' else self.apn)
        self._default = torch.tensor(DEFAULT_SQUASHED, device=device)

        # Running stats for reward normalisation (Welford online estimator).
        # Updated once per learn() call with the full rollout reward batch.
        self.reward_rms = RunningMeanStd()

        # ── Recurrent hidden state ─────────────────────────────────────────────
        # self.h         : the live [num_layers, 1, GRU_HIDDEN_DIM] hidden state
        #                  maintained across select_action() calls within an episode.
        # self._h_buffer : numpy snapshots of self.h captured BEFORE each
        #                  select_action() call.  Consumed by learn() as TBPTT chunk
        #                  initial states; cleared at the end of each learn() call.
        self.h = None
        self._h_buffer = []

        # AMP: mixed-precision scaler for the PPO update backward pass.
        # Only active when CUDA is available; no-ops gracefully on CPU.
        self._amp_enabled = torch.cuda.is_available()
        self._scaler = GradScaler('cuda', enabled=self._amp_enabled)

        # Projection statistics, reset by get_and_reset_proj_stats().
        self._proj_calls = 0
        self._proj_noop = 0
        self._proj_iters = 0

    # ── Episode Management ─────────────────────────────────────────────────────

    def reset(self):
        """Clear per-episode state; call right after env.reset().

        (_h_buffer is NOT cleared here — it spans the full rollout and is only
        consumed inside learn().)
        """
        self.h = None

    def discard_rollout(self):
        """Drop the hidden-state snapshots of steps that will not be learned from."""
        self._h_buffer = []

    def get_and_reset_proj_stats(self):
        """Return projection counters since the last call and reset them.

        Returns:
            A dict with keys:
                - calls: Total projection calls.
                - noop: Calls whose action was already safe (or bypassed).
                - iters: Total ascent steps across all calls.
                - avg_it: Mean ascent steps per call that needed correction.
        """
        calls, noop, iters = self._proj_calls, self._proj_noop, self._proj_iters
        needed = calls - noop
        self._proj_calls = self._proj_noop = self._proj_iters = 0
        return {'calls': calls, 'noop': noop, 'iters': iters,
                'avg_it': (iters / needed) if needed > 0 else 0.0}

    # ── Safety Layer (L2) ──────────────────────────────────────────────────────

    def _project(self, state_t, action):
        """L2: project one step's stage-masked action with the APN (on CPU).

        Args:
            state_t: Observation tensor [1, state_dim].
            action: Squashed, stage-masked action [1, action_dim].

        Returns:
            Projected action [1, action_dim] on the agent's device.
        """
        out, stats = self.apn_infer.project(state_t.cpu(), action.cpu())
        self._proj_calls += 1
        self._proj_noop += int(stats["noop"])
        self._proj_iters += stats["iters"]
        return out.to(state_t.device)

    # ── Action Selection ───────────────────────────────────────────────────────

    def select_action(self, state, intent_noise=0.0):
        """Run the full safe action selection pipeline for one control step.

        Executes Actor -> SERL mask -> APN projection -> SERL mask and records
        the hidden-state snapshot needed by learn() for TBPTT warm-starting.

        The stage mask is enforced at THREE checkpoints so no downstream
        layer can ever produce a structurally illegal action:

        1. Actor samples raw intent z (mask already baked into mean/std).
        2. [SERL checkpoint 2] Masked dims clamped to defaults BEFORE the APN.
        3. APN projects the action inside every constraint head.
        4. [SERL checkpoint 3] Final hard override.

        Args:
            state: Normalised observation of shape [state_dim].
            intent_noise: Relative Gaussian noise on the squashed intent before
                the safety stack (evaluation robustness tests); 0 disables it.

        Returns:
            A tuple of:
                - final_action: Projected, stage-masked action in [-1, 1],
                  shape [action_dim].
                - raw_intent: Unmodified pre-tanh z, shape [action_dim]; stored
                  in memory for PPO log-prob computation in learn().
                - log_prob: Log pi_old(z | s, h) at collection time (float).
                - state_in: The state fed to the actor, shape [state_dim].
        """
        with torch.no_grad():
            state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), device=device).unsqueeze(0)

            # A. Actor: sample raw intent with recurrent memory.  Snapshot h
            # BEFORE this step so learn() can use it as a TBPTT chunk start.
            h_snap = (self.h.cpu().numpy() if self.h is not None
                      else np.zeros((GRU_NUM_LAYERS, 1, GRU_HIDDEN_DIM), dtype=np.float32))
            self._h_buffer.append(h_snap)

            raw_intent, log_prob, self.h = self.policy_old.act(state_t, self.h)
            squashed = torch.tanh(raw_intent)

            if intent_noise > 0:
                intent = squashed.cpu().numpy()
                intent = np.clip(intent + intent * np.random.normal(0, intent_noise, size=intent.shape),
                                 -1.0, 1.0)
                squashed = torch.as_tensor(intent, dtype=torch.float32, device=device)

            # ── SERL checkpoint 2 ─────────────────────────────────────────────
            if self.use_safeguard:
                mask = PhycocyaninEnvCore.get_action_mask(state_t)
                squashed = squashed * mask + self._default * (1 - mask)

            # ── L2: APN projection ────────────────────────────────────────────
            if self.use_safeguard and self.use_projection and self.apn is not None:
                final_action = self._project(state_t, squashed)
            else:
                final_action = squashed

            # ── SERL checkpoint 3 (final absolute override) ───────────────────
            if self.use_safeguard:
                final_action = final_action * mask + self._default * (1 - mask)

        return (final_action.cpu().numpy()[0], raw_intent.cpu().numpy()[0],
                log_prob.item(), state_t.cpu().numpy()[0])

    # ── PPO Update Internals ───────────────────────────────────────────────────

    def _prepare_tensors(self, memory):
        """Materialise rollout memory as device tensors.

        Returns:
            A tuple of (old_states [T, D], old_latents [T, A], old_logprobs [T],
            old_safe_acts [T, A], done_t [T]).
        """
        n = memory.size
        return (torch.from_numpy(memory.states[:n]).to(device),
                torch.from_numpy(memory.actions[:n]).to(device),
                torch.from_numpy(memory.logprobs[:n]).to(device),
                torch.from_numpy(memory.safe_actions[:n]).to(device),
                torch.from_numpy(memory.is_terminals[:n].astype(np.float32)).to(device))

    def _compute_advantages(self, old_states, done_t, rewards, lam):
        """Compute Generalised Advantage Estimation (GAE) with recurrent value estimates.

        Args:
            old_states: State tensor of shape [T, state_dim].
            done_t: Terminal flag tensor of shape [T].
            rewards: Reward array of shape [T].
            lam: GAE lambda for bias-variance trade-off.

        Returns:
            A tuple of:
                - b_advs: Normalised advantage tensor of shape [T].
                - b_returns: Return targets of shape [T].
                - values_t: Raw value estimates of shape [T].
        """
        T = old_states.shape[0]
        with torch.no_grad():
            feat_v, _ = self.policy._encode_sequence(
                old_states.unsqueeze(0), self.policy.init_hidden(1), done_t.unsqueeze(0))
            values_t = self.policy.critic(feat_v).reshape(T)
            values = values_t.cpu().numpy()
            values_padded = np.append(values, 0.0)

            self.reward_rms.update(rewards)
            rewards_n = self.reward_rms.normalize(rewards)
            not_done = 1.0 - done_t.cpu().numpy()

            advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                delta = (rewards_n[t] + self.gamma * values_padded[t + 1] * not_done[t]
                         - values_padded[t])
                gae = delta + self.gamma * lam * not_done[t] * gae
                advantages[t] = gae

            b_advs = torch.from_numpy(advantages).to(device)
            b_returns = b_advs + values_t
            b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

        return b_advs, b_returns, values_t

    def _build_tbptt_chunks(self, *seqs):
        """Cut [T, ...] rollout tensors into [n_chunks, L, ...] TBPTT chunks.

        L is SEQ_LEN (the whole rollout if shorter); a trailing remainder
        shorter than L is dropped.  Chunk i starts from the hidden state
        recorded at its first step.

        Args:
            *seqs: Tensors whose first dimension is time.

        Returns:
            A tuple of (h_init [num_layers, n_chunks, GRU_HIDDEN_DIM], chunked
            tensors in the order given).
        """
        T = seqs[0].shape[0]
        L = min(SEQ_LEN, T)
        n = T // L

        zeros = np.zeros((GRU_NUM_LAYERS, 1, GRU_HIDDEN_DIM), dtype=np.float32)
        h_init = np.concatenate(
            [self._h_buffer[c * L] if c * L < len(self._h_buffer) else zeros for c in range(n)], axis=1)
        h_init = torch.from_numpy(h_init).to(device)

        chunks = tuple(x[:n * L].reshape(n, L, *x.shape[1:]) for x in seqs)
        return (h_init,) + chunks

    def _safe_mean(self, c_states, mean):
        """The actor's mean action as select_action() would send it before projection.

        Squashed with tanh and stage-masked to the same defaults, so the APN
        sees the actor's own deterministic intent.  Differentiable in the
        actor's parameters.

        Args:
            c_states: State chunk [B, L, state_dim].
            mean: Actor mean [B, L, action_dim] from evaluate().

        Returns:
            Tensor [B, L, action_dim] (float32).
        """
        a = torch.tanh(mean.float())
        mask = PhycocyaninEnvCore.get_action_mask(c_states)
        return a * mask + self._default * (1 - mask)

    def _compute_mapping_penalty(self, c_states, safe_mean, c_safe_acts):
        """Pull the actor's mean towards the projected action it led to.

        Weighted by how far the mean action is outside the APN's acceptance
        margin, so corrected steps dominate; zero where the APN is bypassed
        (Harvesting, Idle).  Only meaningful with the projection on.

        Args:
            c_states: State chunk of shape [B, L, state_dim].
            safe_mean: Actor mean action from _safe_mean(), [B, L, action_dim].
            c_safe_acts: Projected actions sent to the plant, [B, L, action_dim].

        Returns:
            Scalar mapping penalty loss (0.0 without projection).
        """
        if not (self.use_safeguard and self.use_projection and self.use_safety_grad
                and self.apn is not None):
            return 0.0
        with torch.no_grad(), autocast(device_type='cuda', enabled=False):
            m = self.apn.margins(c_states.float(), safe_mean.detach())          # [B, L, H]
            weight = torch.clamp(SAFE_LOGIT - m.min(dim=-1).values, min=0.0) + 0.1
            weight = weight * projection_active(c_states).float()             # [B, L]
        per_sample = (safe_mean - c_safe_acts).pow(2).mean(dim=-1)            # [B, L]
        return (per_sample * weight).mean()

    def _compute_safety_loss(self, c_states, safe_mean):
        """The actor's safety gradient: the APN's hinge on its own mean action.

        Backpropagates every violated head's margin gradient through the
        frozen APN into the actor, so the actor learns to propose actions the
        projection no longer has to correct.

        Args:
            c_states: State chunk [B, L, state_dim].
            safe_mean: Actor mean action from _safe_mean(), [B, L, action_dim].

        Returns:
            Scalar loss (0.0 when the safety gradient is off).
        """
        if not (self.use_safeguard and self.use_safety_grad and self.apn is not None):
            return 0.0
        with autocast(device_type='cuda', enabled=False):
            return self.apn.safety_loss(c_states.float(), safe_mean)

    def _compute_loss(self, batch, dist_entropy, logprobs, state_values, mean):
        """Compute the composite PPO loss for one minibatch of TBPTT chunks.

        Args:
            batch: Tuple of (c_states, c_advs, c_rets, c_old_lp, c_safe_acts,
                c_old_vals), each [B, L, ...].
            dist_entropy: Entropy tensor of shape [B, L].
            logprobs: Log-probability tensor of shape [B, L].
            state_values: Value estimates of shape [B, L, 1].
            mean: Actor mean of shape [B, L, action_dim].

        Returns:
            Scalar composite loss combining surrogate, value, entropy, mapping
            penalty and safety-gradient terms.
        """
        c_states, c_advs, c_rets, c_old_lp, c_safe_acts, c_old_vals = batch

        safe_mean = self._safe_mean(c_states, mean)
        mapping_penalty = self._compute_mapping_penalty(c_states, safe_mean, c_safe_acts)
        safety_loss = self._compute_safety_loss(c_states, safe_mean)

        ratios = torch.exp(logprobs - c_old_lp)
        surr1 = ratios * c_advs
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * c_advs

        sv = state_values.squeeze(-1)
        v_clipped = c_old_vals + torch.clamp(sv - c_old_vals, -self.eps_clip, self.eps_clip)
        value_loss = torch.max((sv - c_rets).pow(2), (v_clipped - c_rets).pow(2)).mean()

        return (
            -torch.min(surr1, surr2).mean()
            + 0.5 * value_loss
            - self.entropy_coeff * dist_entropy.mean()
            + MAPPING_COEF * mapping_penalty
            + SAFETY_GRAD_COEF * safety_loss
        )

    # ── Main Learning Step ─────────────────────────────────────────────────────

    def learn(self, memory, lam=0.95):
        """Recurrent PPO update (Deep Recurrent Policy Optimisation).

        Executes the full update pipeline in modular stages:
            1. Tensor materialisation from rollout memory.
            2. Advantage estimation (GAE) with the recurrent value network.
            3. Sequence chunking for TBPTT.
            4. SGD epoch loop with clipped surrogate + value + entropy +
               mapping penalty + safety gradient.
            5. Policy synchronisation and state cleanup.

        Args:
            memory: PPOMemory instance containing the collected rollout.
            lam: GAE lambda for bias-variance trade-off.
        """
        # Step 1: Prepare Tensors
        old_states, old_latents, old_logprobs, old_safe_acts, done_t = self._prepare_tensors(memory)

        # Step 2: GAE Computation
        b_advs, b_returns, values_t = self._compute_advantages(
            old_states, done_t, memory.rewards[:memory.size], lam)

        # Step 3: TBPTT Chunking
        (h_init, c_states, c_latents, c_dones, c_advs, c_rets, c_old_lp, c_safe_acts,
         c_old_vals) = self._build_tbptt_chunks(
            old_states, old_latents, done_t, b_advs, b_returns, old_logprobs,
            old_safe_acts, values_t)

        # Step 4: SGD Epochs over minibatches of chunks
        n_chunks = c_states.shape[0]
        for _ in range(self.K_epochs):
            for idx in torch.randperm(n_chunks, device=device).split(CHUNKS_PER_BATCH):
                batch = (c_states[idx], c_advs[idx], c_rets[idx], c_old_lp[idx],
                         c_safe_acts[idx], c_old_vals[idx])

                with autocast(device_type='cuda', enabled=self._amp_enabled):
                    logprobs, state_values, dist_entropy, mean = self.policy.evaluate(
                        c_states[idx], c_latents[idx], h_init[:, idx], c_dones[idx])
                    loss = self._compute_loss(batch, dist_entropy, logprobs, state_values, mean)

                # Gradient step
                self.optimizer.zero_grad()
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self._scaler.step(self.optimizer)
                self._scaler.update()

            if any(torch.isnan(p).any() for p in self.policy.parameters()):
                self.policy.load_state_dict(self.policy_old.state_dict())
                break

        # Step 5: Sync old policy & Reset State
        self.policy_old.load_state_dict(self.policy.state_dict())
        self._h_buffer = []
        self.h = None
