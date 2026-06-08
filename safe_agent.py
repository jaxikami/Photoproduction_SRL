"""
Safe Reinforcement Learning Agent using Action Projection.

This module implements a Safe Proximal Policy Optimization (PPO) agent.
It uses an Action Projection Network (APN) to safeguard the agent's actions
by projecting unsafe actions onto a learned safety manifold before they are
applied to the environment.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os
from numba import njit
from pretrain import ActionProjectionNetwork

# Hardware acceleration setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disable CuDNN and MIOpen backends to avoid ROCm GRU bugs (miopenStatusUnknownError)
torch.backends.cudnn.enabled = False
if hasattr(torch.backends, 'miopen'):
    torch.backends.miopen.enabled = False


@njit(cache=True)
def _discount_rewards(raw_rewards, terminals, gamma):
    """Numba-accelerated GAE-style reward discounting."""
    n = len(raw_rewards)
    discounted = np.zeros(n, dtype=np.float32)
    running = 0.0
    for i in range(n - 1, -1, -1):
        if terminals[i]:
            running = 0.0
        running = raw_rewards[i] + gamma * running
        discounted[i] = running
    return discounted


# =============================================================================
# GRU + Skip-Connection Actor-Critic Architecture
# =============================================================================

class ActorCritic(nn.Module):
    """Safe-agent Actor-Critic with GRU temporal encoder and skip-connection.

    The GRU provides temporal memory so the agent can anticipate constraint
    violations (especially g2 ratio buildup) before they occur. A skip
    connection concatenates GRU output with the raw state for both heads.

    Action masking follows the established protocol:
        - Masked dims get a fixed default intent (pre-Tanh)
        - Masked dims get near-zero std to suppress exploration
        - Log-probs exclude masked dims
    """
    def __init__(self, state_dim=12, action_dim=4, hidden_dim=128):
        """Initializes the GRU encoder and Actor/Critic heads.

        Args:
            state_dim (int, optional): Observation dimension. Defaults to 12.
            action_dim (int, optional): Action dimension. Defaults to 4.
            hidden_dim (int, optional): GRU hidden state dimension. Defaults to 128.
        """
        super(ActorCritic, self).__init__()
        self.LOG_STD_MIN = -1.0
        self.LOG_STD_MAX = 0.5
        self.env_state_dim = state_dim
        self.hidden_dim = hidden_dim

        # GRU temporal encoder
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=False)

        # Actor head: GRU features + skip connection from raw state
        actor_input_dim = hidden_dim + state_dim
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, 256), nn.ELU(),
            nn.Linear(256, 256),             nn.ELU(),
            nn.Linear(256, action_dim)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        # Critic head: GRU features + skip connection from raw state
        self.critic = nn.Sequential(
            nn.Linear(actor_input_dim, 256), nn.ELU(),
            nn.Linear(256, 256),             nn.ELU(),
            nn.Linear(256, 1)
        )

    def init_hidden(self, batch_size=1):
        """Creates a zero-initialized GRU hidden state.

        Args:
            batch_size (int): Batch dimension for the hidden state.

        Returns:
            torch.Tensor: Zero tensor of shape (1, batch_size, hidden_dim).
        """
        return torch.zeros(1, batch_size, self.hidden_dim, device=next(self.parameters()).device)

    def act(self, state, hidden=None):
        """Generates an action using GRU temporal encoding and stage-aware masking.

        Args:
            state (torch.Tensor): Current state tensor of shape (1, state_dim).
            hidden (torch.Tensor or None): GRU hidden state from previous step.

        Returns:
            tuple:
                - z (torch.Tensor): Squashed action [-1, 1]
                - log_prob (torch.Tensor): Log probability of chosen action.
                - z_raw (torch.Tensor): Unbounded Gaussian sample.
                - hidden_new (torch.Tensor): Updated GRU hidden state.
                - masked_mean (torch.Tensor): Distribution mean (for log-prob reuse).
                - masked_std (torch.Tensor): Distribution std (for log-prob reuse).
                - mask (torch.Tensor): Stage action mask.
        """
        from env_core import PhycocyaninEnvCore

        if hidden is None:
            hidden = self.init_hidden(batch_size=state.shape[0])

        # GRU forward: state shape (1, batch, state_dim) -> gru_out (1, batch, hidden_dim)
        gru_out, hidden_new = self.gru(state.unsqueeze(0), hidden)
        gru_out = gru_out.squeeze(0)  # (batch, hidden_dim)

        # Skip connection: concatenate GRU features with raw state
        features = torch.cat([gru_out, state], dim=-1)  # (batch, hidden_dim + state_dim)
        features = torch.nan_to_num(features, nan=0.0)

        mean = self.actor(features)
        mean = torch.nan_to_num(mean, nan=0.0)
        std  = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))

        # Stage mask from observation
        mask = PhycocyaninEnvCore.get_action_mask(state[..., :self.env_state_dim])

        # Default intents for masked dims (pre-Tanh values):
        # time_mult → tanh(0) = 0.0 → maps to 1.0 multiplier (neutral)
        # I → tanh(-10) ≈ -1.0 → maps to I_MIN
        # Fn → tanh(-10) ≈ -1.0 → maps to 0
        # Fout → tanh(-10) ≈ -1.0 → maps to 0
        default_intent = torch.full_like(mean, -10.0)
        default_intent[..., 0] = -0.4329  # time_mult default: neutral 1.0 multiplier

        masked_mean = mean * mask + default_intent * (1 - mask)
        masked_std  = std * mask + 1e-8 * (1 - mask)

        dist  = Normal(masked_mean, masked_std)
        z_raw = dist.sample()
        z     = torch.tanh(z_raw)

        log_prob = (dist.log_prob(z_raw) * mask).sum(dim=-1)

        return z.detach(), log_prob.detach(), z_raw.detach(), hidden_new.detach(), masked_mean.detach(), masked_std.detach(), mask.detach()

    def evaluate(self, state, z_raw, is_terminals=None):
        """Re-evaluates stored intents during the PPO update with GRU context.

        Uses chunked GRU processing for speed: processes contiguous segments
        between episode boundaries as single batched sequences, avoiding the
        overhead of per-timestep Python loops.

        Args:
            state (torch.Tensor): Batch of state tensors (T, state_dim).
            z_raw (torch.Tensor): Batch of unbounded actions (T, action_dim).
            is_terminals (torch.Tensor or None): Episode boundary flags (T,).

        Returns:
            tuple: (log_probs, state_values, dist_entropy)
        """
        features = self.compute_features(state, is_terminals)
        return self.evaluate_from_features(features, state, z_raw)

    def compute_features(self, state, is_terminals=None):
        """Runs the GRU over full sequences to produce temporal features.

        Args:
            state (torch.Tensor): Batch of state tensors (T, state_dim).
            is_terminals (torch.Tensor or None): Episode boundary flags (T,).

        Returns:
            torch.Tensor: Feature tensor of shape (T, hidden_dim + state_dim).
        """
        T = state.shape[0]

        if is_terminals is not None:
            boundary_indices = torch.where(is_terminals)[0].tolist()
        else:
            boundary_indices = []

        segments = []
        start = 0
        for end_idx in boundary_indices:
            segments.append((start, end_idx + 1))
            start = end_idx + 1
        if start < T:
            segments.append((start, T))

        features = torch.empty(T, self.hidden_dim + state.shape[-1], device=state.device)
        for seg_start, seg_end in segments:
            seg_states = state[seg_start:seg_end].unsqueeze(1)
            hidden = self.init_hidden(batch_size=1)
            gru_out, _ = self.gru(seg_states, hidden)
            gru_out = gru_out.squeeze(1)
            features[seg_start:seg_end] = torch.cat(
                [gru_out, state[seg_start:seg_end]], dim=-1)

        features = torch.nan_to_num(features, nan=0.0)
        return features

    def evaluate_from_features(self, features, state, z_raw):
        """Evaluates log-probs, values, entropy from pre-computed GRU features.

        Args:
            features (torch.Tensor): (T, hidden_dim + state_dim) from compute_features.
            state (torch.Tensor): Batch of state tensors (T, state_dim) for masking.
            z_raw (torch.Tensor): Batch of unbounded actions (T, action_dim).

        Returns:
            tuple: (log_probs, state_values, dist_entropy)
        """
        from env_core import PhycocyaninEnvCore

        mean = self.actor(features)
        mean = torch.nan_to_num(mean, nan=0.0)
        std  = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))

        mask = PhycocyaninEnvCore.get_action_mask(state[..., :self.env_state_dim])

        default_intent = torch.full_like(mean, -10.0)
        default_intent[..., 0] = -0.4329
        masked_mean = mean * mask + default_intent * (1 - mask)
        masked_std  = std * mask + 1e-8 * (1 - mask)

        dist = Normal(masked_mean, masked_std)
        log_probs    = (dist.log_prob(z_raw) * mask).sum(dim=-1)
        dist_entropy = (dist.entropy() * mask).sum(dim=-1)

        state_values = self.critic(features)
        return log_probs, state_values, dist_entropy

    def evaluate_single(self, state, z_raw, hidden=None):
        """Evaluates one state-action pair under a provided GRU hidden state.

        This is used during rollout to compute a behavior-consistent log-prob
        for the actually executed (projected) action.

        Args:
            state (torch.Tensor): Tensor of shape (1, state_dim).
            z_raw (torch.Tensor): Tensor of shape (1, action_dim), pre-tanh.
            hidden (torch.Tensor or None): GRU hidden state at this timestep.

        Returns:
            tuple: (log_prob, state_value, dist_entropy, hidden_new)
        """
        from env_core import PhycocyaninEnvCore

        if hidden is None:
            hidden = self.init_hidden(batch_size=state.shape[0])

        gru_out, hidden_new = self.gru(state.unsqueeze(0), hidden)
        gru_out = gru_out.squeeze(0)
        features = torch.cat([gru_out, state], dim=-1)

        mean = self.actor(features)
        std  = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))

        mask = PhycocyaninEnvCore.get_action_mask(state[..., :self.env_state_dim])
        default_intent = torch.full_like(mean, -10.0)
        default_intent[..., 0] = -0.4329
        masked_mean = mean * mask + default_intent * (1 - mask)
        masked_std  = std * mask + 1e-8 * (1 - mask)

        dist = Normal(masked_mean, masked_std)
        log_prob = (dist.log_prob(z_raw) * mask).sum(dim=-1)
        dist_entropy = (dist.entropy() * mask).sum(dim=-1)
        state_value = self.critic(features)
        return log_prob, state_value, dist_entropy, hidden_new


class SPRL_Agent:
    """Safe PPO agent combining a GRU ActorCritic, stage masking, and APN projection."""
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coeff):
        self.gamma          = gamma
        self.eps_clip       = eps_clip
        self.K_epochs       = K_epochs
        self.entropy_coeff  = entropy_coeff

        self._hidden = None

        # Actor-Critic
        self.policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.gru.parameters(),          'lr': lr_actor},
            {'params': self.policy.actor.parameters(),        'lr': lr_actor},
            {'params': [self.policy.log_std],                 'lr': lr_actor},
            {'params': self.policy.critic.parameters(),       'lr': lr_critic, 'weight_decay': 1e-5}
        ])

        self.policy_old = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Safeguard (APN)
        _ckpt_paths = [
            os.path.join("policy", "action_projection_network.pth"),
            "action_projection_network.pth",
        ]
        for _ckpt in _ckpt_paths:
            if os.path.exists(_ckpt):
                self.safeguard = ActionProjectionNetwork.from_checkpoint(
                    _ckpt, device, state_dim=state_dim, action_dim=action_dim)
                self.safeguard.eval()
                for p in self.safeguard.parameters():
                    p.requires_grad_(False)
                break
        else:
            print("[Safeguard] WARNING: APN not found. Actions will be unfiltered.")
            self.safeguard = ActionProjectionNetwork(
                state_dim=state_dim, action_dim=action_dim).to(device)

        self._proj_calls = 0
        self._proj_noop  = 0
        self._proj_iters = 0

        # Pre-allocate constant tensors for select_action hot path
        self._default_squashed = torch.tensor([-0.407, -1.0, -1.0, -1.0], device=device)
        self._state_buffer = torch.zeros(1, state_dim, device=device)  # Reusable input buffer

        self.MseLoss = nn.MSELoss()

    def reset_hidden(self):
        """Reset GRU hidden state at episode start."""
        self._hidden = None

    def get_and_reset_proj_stats(self):
        calls  = self._proj_calls
        noop   = self._proj_noop
        iters  = self._proj_iters
        needed = calls - noop
        avg_it = (iters / needed) if needed > 0 else 0.0
        self._proj_calls = self._proj_noop = self._proj_iters = 0
        return {'calls': calls, 'noop': noop, 'iters': iters, 'avg_it': avg_it}

    def _project_to_safe(self, state_norm, action, lr=0.6, max_steps=8, threshold=0.711):
        """Gradient-ascend the APN margin surface to find a safe action proxy.

        If the initially proposed action is unsafe, this method performs gradient
        ascent on the action space to maximize the APN safety classification score.
        Stage-masked dimensions are zeroed in the gradient to prevent modifying
        locked channels. Uses a constant step size with mild decay.

        Args:
            state_norm (torch.Tensor): Observation tensor.
            action (torch.Tensor): Initial proposed action tensor.
            lr (float, optional): Gradient ascent learning rate. Defaults to 0.25.
            max_steps (int, optional): Maximum optimization steps. Defaults to 15.
            threshold (float, optional): Target safety probability. Defaults to 0.95.

        Returns:
            torch.Tensor: The projected, safe action (or closest proxy).
        """
        from env_core import PhycocyaninEnvCore

        a = action.clone().detach()
        state_fixed = state_norm.detach()
        self._proj_calls += 1

        # Stage mask for gradient zeroing
        stage_mask = PhycocyaninEnvCore.get_action_mask(state_fixed)

        # Bypass projection in Stage 2 (Harvesting) and Stage 3 (Idle) to avoid phantom gradients
        # lock-in during draining and shutdown.
        is_stage_2_or_3 = (state_fixed[..., 6] > 0.5) | (state_fixed[..., 7] > 0.5)
        if is_stage_2_or_3.any():
            self._proj_noop += 1
            return a

        with torch.no_grad():
            p = self.safeguard.classify(state_fixed, a)
            if p.item() >= threshold:
                self._proj_noop += 1
                return a

        best_a      = a.clone()
        best_margin = self.safeguard(state_fixed, a).item()

        with torch.enable_grad():
            for step in range(max_steps):
                self._proj_iters += 1
                a_var  = a.clone().requires_grad_(True)
                margin = self.safeguard(state_fixed, a_var)

                p = torch.sigmoid(margin)
                if p.item() >= threshold:
                    return a_var.detach()

                # Track best iterate seen
                m_val = margin.item()
                if m_val > best_margin:
                    best_margin = m_val
                    best_a      = a_var.detach().clone()

                grad = torch.autograd.grad(margin.sum(), a_var)[0]

                with torch.no_grad():
                    grad = grad.clone()
                    # Zero gradient for masked dimensions
                    grad = grad * stage_mask

                    at_lower = (a_var.data <= -0.9999) & (grad < 0)
                    at_upper = (a_var.data >=  0.9999) & (grad > 0)
                    grad[at_lower | at_upper] = 0.0

                    # Constant step size with mild decay
                    step_size = lr / (1.0 + step * 0.03)
                    a = a + step_size * grad.sign()
                    a = a.clamp(-1.0, 1.0)

        return best_a

    def select_action(self, state_norm):
        """Selects a safe action by projecting the raw actor output through the APN.

        Full action pipeline: Actor -> Stage Mask -> APN Projection -> Stage Mask.
        This ensures the final action is safe while strictly respecting process stages.

        Args:
            state_norm (np.ndarray): Normalized observation array.

        Returns:
            tuple:
                - u_safe_np (np.ndarray): Executed safe action.
                - log_prob (np.ndarray): Log probability of the raw intent.
                - z_raw (np.ndarray): Unbounded intent pre-projection.
        """

        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm).to(device).unsqueeze(0)

            # Generate intent via GRU actor (mask baked into act())
            z, log_prob, z_raw, self._hidden, dist_mean, dist_std, mask = self.policy_old.act(state_t, self._hidden)

            # Apply SERL mask checkpoint 2 (pre-APN)
            z_masked = z * mask + self._default_squashed * (1 - mask)

        # APN gradient projection
        u_safe = self._project_to_safe(state_t, z_masked)

        # Final SERL mask checkpoint 3 (post-APN)
        with torch.no_grad():
            u_safe = u_safe * mask + self._default_squashed * (1 - mask)

        # Convert executed squashed action to pre-tanh space for PPO storage.
        # This keeps memory.raw_actions aligned with what was executed in env.
        u_safe_np = u_safe.cpu().numpy().flatten()
        u_safe_raw_np = np.arctanh(np.clip(u_safe_np, -0.999999, 0.999999))

        # Compute log-prob for projected action analytically from cached distribution.
        # This avoids a redundant GRU forward pass — same result as evaluate_single.
        with torch.no_grad():
            u_safe_raw_t = torch.FloatTensor(u_safe_raw_np).to(device).unsqueeze(0)
            dist = Normal(dist_mean, dist_std)
            log_prob_exec = (dist.log_prob(u_safe_raw_t) * mask).sum(dim=-1)

        return (u_safe_np,
                log_prob_exec.cpu().numpy(),
                u_safe_raw_np)

    def learn(self, memory):
        """Executes the PPO update with chunked GRU processing.

        Args:
            memory (Memory): Buffer containing collected trajectories.
        """
        # Fast vectorized reward discounting using numba
        if hasattr(memory, '_rewards'):
            # Fast path: use pre-allocated numpy arrays directly
            raw_rewards = memory._rewards[:memory._ptr].copy()
            terminals = memory._is_terminals[:memory._ptr].astype(np.float32)
        else:
            raw_rewards = np.array(memory.rewards, dtype=np.float32)
            terminals = np.array(memory.is_terminals, dtype=np.float32)
        discounted = _discount_rewards(raw_rewards, terminals, self.gamma)

        rewards = torch.from_numpy(discounted).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Use fast pre-stacked tensor path if available
        if hasattr(memory, 'get_tensors'):
            old_states, _, old_z_raw, old_logprobs, is_terminals = memory.get_tensors(device)
        else:
            old_states    = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
            old_z_raw     = torch.squeeze(torch.stack(memory.raw_actions, dim=0)).detach().to(device)
            old_logprobs  = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)
            is_terminals  = torch.tensor(memory.is_terminals, dtype=torch.bool).to(device)

        # Pre-compute fixed advantages once (stable target across all epochs, like Standard RL)
        with torch.no_grad():
            _, old_state_values, _ = self.policy.evaluate(
                old_states, old_z_raw, is_terminals=is_terminals)
            advantages = rewards - old_state_values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size     = old_states.size(0)
        mini_batch_size = 256

        for _ in range(self.K_epochs):
            # GRU features computed once per epoch then detached — temporal context
            # is preserved but actor/critic get fresh forward passes per mini-batch
            # (matching the fermentation PPO update mechanism exactly).
            with torch.no_grad():
                features_all = self.policy.compute_features(
                    old_states, is_terminals=is_terminals)

            permutation = torch.randperm(batch_size)
            for start_idx in range(0, batch_size, mini_batch_size):
                batch_indices = permutation[start_idx : start_idx + mini_batch_size]

                b_features = features_all[batch_indices]
                b_states   = old_states[batch_indices]
                b_z_raw    = old_z_raw[batch_indices]
                b_old_lp   = old_logprobs[batch_indices]
                b_rewards  = rewards[batch_indices]
                b_adv      = advantages[batch_indices]

                logprobs, state_values, dist_entropy = self.policy.evaluate_from_features(
                    b_features, b_states, b_z_raw)

                ratios = torch.exp(logprobs - b_old_lp)
                surr1  = ratios * b_adv
                surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                ppo_loss = -torch.min(surr1, surr2).mean()

                loss = (ppo_loss
                        + 0.5  * self.MseLoss(state_values.squeeze(), b_rewards)
                        - self.entropy_coeff * dist_entropy.mean())

                if torch.isnan(loss) or torch.isinf(loss):
                    self.policy.load_state_dict(self.policy_old.state_dict())
                    self.policy_old.load_state_dict(self.policy.state_dict())
                    return

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                if any(torch.isnan(p).any() for p in self.policy.parameters()):
                    self.policy.load_state_dict(self.policy_old.state_dict())
                    self.policy_old.load_state_dict(self.policy.state_dict())
                    return

        self.policy_old.load_state_dict(self.policy.state_dict())
