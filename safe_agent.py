import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os

# Hardware acceleration setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActionProjectionNetwork(nn.Module):
    """
    Safety Proximity Regressor for the multi-stage photobioreactor.

    Given a (normalised-state, action) pair, outputs a continuous signed
    margin score:
      margin > 0  →  action is safe
      margin < 0  →  action is unsafe
      margin = 0  →  action is on the safety boundary

    Constraints covered: G1, G2, G4 (overflow).
    G3 (terminal nitrate) is handled by the GRU temporal context and
    Lagrangian multiplier.

    Architecture:
      Linear(15→128) → LayerNorm → Mish
      → 3× Residual blocks (128→128)
      → Linear(128→1)  [raw margin]
      + 3 per-constraint auxiliary heads (train-only)
    """
    def __init__(self, state_dim: int = 11, action_dim: int = 4,
                 latent_dim: int = 128):
        super(ActionProjectionNetwork, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish()
        )

        # Three residual blocks for depth
        self.res_block1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.res_block3 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.res_act = nn.Mish()

        self.margin_head = nn.Linear(latent_dim, 1)

        # Per-constraint auxiliary heads (train-only, direct gradient signal)
        self.g1_head = nn.Linear(latent_dim, 1)
        self.g2_head = nn.Linear(latent_dim, 1)
        self.g4_head = nn.Linear(latent_dim, 1)

        nn.init.zeros_(self.margin_head.weight)
        nn.init.zeros_(self.margin_head.bias)

    def forward(self, state_norm, action_norm):
        x_in   = torch.cat([state_norm, action_norm], dim=-1)
        x_enc  = self.encoder(x_in)
        x_enc  = self.res_act(x_enc + self.res_block1(x_enc))
        x_enc  = self.res_act(x_enc + self.res_block2(x_enc))
        x_enc  = self.res_act(x_enc + self.res_block3(x_enc))
        margin = self.margin_head(x_enc).squeeze(-1)
        return margin

    def classify(self, state_norm, action_norm):
        return torch.sigmoid(self.forward(state_norm, action_norm))


# =============================================================================
# GRU + Skip-Connection Actor-Critic Architecture
# =============================================================================

class StateEncoderGRU(nn.Module):
    """Temporal context encoder with GRU."""
    def __init__(self, state_dim=11, embed_dim=32, gru_hidden=64, gru_layers=1):
        super(StateEncoderGRU, self).__init__()
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        self.embed = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ELU(),
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )

    def forward(self, state, hidden=None):
        emb = self.embed(state)
        emb_seq = emb.unsqueeze(1)
        gru_out, hidden = self.gru(emb_seq, hidden)
        gru_out = gru_out.squeeze(1)
        return gru_out, hidden


class StateEncoderSkip(nn.Module):
    """Instantaneous state encoder (skip connection)."""
    def __init__(self, state_dim=11, skip_dim=32):
        super(StateEncoderSkip, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(state_dim, skip_dim),
            nn.ELU(),
        )

    def forward(self, state):
        return self.encode(state)


class ActorCritic(nn.Module):
    """
    Safe-agent Actor-Critic with dual-encoder (GRU + skip) and
    stage-aware action masking for the 4D action space.

    Action masking follows the Fermentation-PPO pattern:
      - Masked dims get a fixed default intent (pre-Tanh)
      - Masked dims get near-zero std to suppress exploration
      - Log-probs exclude masked dims
    """
    def __init__(self, state_dim=11, action_dim=4,
                 embed_dim=32, gru_hidden=64, skip_dim=32):
        super(ActorCritic, self).__init__()
        self.LOG_STD_MIN = -1.0
        self.LOG_STD_MAX = 0.5
        self.env_state_dim = state_dim

        # Dual Encoders
        self.gru_encoder  = StateEncoderGRU(state_dim, embed_dim, gru_hidden)
        self.skip_encoder = StateEncoderSkip(state_dim, skip_dim)

        fused_dim = gru_hidden + skip_dim

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ELU(),
            nn.Linear(128, 64),         nn.ELU(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ELU(),
            nn.Linear(128, 64),         nn.ELU(),
            nn.Linear(64, 1)
        )

    def _encode(self, state, hidden=None):
        gru_out, hidden = self.gru_encoder(state, hidden)
        skip            = self.skip_encoder(state)
        fused           = torch.cat([gru_out, skip], dim=-1)
        return fused, hidden

    def act(self, state, hidden=None):
        """
        Generates an action with stage-aware masking.

        Returns:
            z        : squashed action [-1, 1]
            log_prob : log π(z | s)
            z_raw    : unbounded Gaussian sample
            hidden   : updated GRU hidden state
        """
        from env_core import PhycocyaninEnvCore

        fused, hidden = self._encode(state, hidden)
        mean = self.actor(fused)
        std  = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))

        # Stage mask from observation
        mask = PhycocyaninEnvCore.get_action_mask(state[..., :self.env_state_dim])

        # Default intents for masked dims (pre-Tanh values):
        # time_mult → tanh(-0.35) ≈ -0.34 → maps to ~0.5 multiplier
        # I → tanh(-10) ≈ -1.0 → maps to I_MIN
        # Fn → tanh(-10) ≈ -1.0 → maps to 0
        # Fout → tanh(-10) ≈ -1.0 → maps to 0
        default_intent = torch.full_like(mean, -10.0)
        default_intent[..., 0] = -0.34657359  # time_mult default

        masked_mean = mean * mask + default_intent * (1 - mask)
        masked_std  = std * mask + 1e-8 * (1 - mask)

        dist  = Normal(masked_mean, masked_std)
        z_raw = dist.sample()
        z     = torch.tanh(z_raw)

        log_prob = (dist.log_prob(z_raw) * mask).sum(dim=-1)

        return z.detach(), log_prob.detach(), z_raw.detach(), hidden

    def evaluate(self, state, z_raw):
        """
        Re-evaluates stored intents during the PPO update.
        Hidden state is not threaded (standard PPO practice).
        """
        from env_core import PhycocyaninEnvCore

        fused, _ = self._encode(state, hidden=None)
        mean     = self.actor(fused)
        std      = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))

        mask = PhycocyaninEnvCore.get_action_mask(state[..., :self.env_state_dim])

        default_intent = torch.full_like(mean, -10.0)
        default_intent[..., 0] = -0.34657359
        masked_mean = mean * mask + default_intent * (1 - mask)
        masked_std  = std * mask + 1e-8 * (1 - mask)

        dist = Normal(masked_mean, masked_std)

        log_probs    = (dist.log_prob(z_raw) * mask).sum(dim=-1)
        dist_entropy = (dist.entropy() * mask).sum(dim=-1)
        state_values = self.critic(fused)
        return log_probs, state_values, dist_entropy


class SPRL_Agent:
    """
    Safe PPO agent with dual-encoder ActorCritic, stage-aware action masking,
    and APN gradient projection safety filter.
    """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coeff):
        self.gamma          = gamma
        self.eps_clip       = eps_clip
        self.K_epochs       = K_epochs
        self.entropy_coeff  = entropy_coeff

        self._hidden = None

        # Actor-Critic
        self.policy = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.gru_encoder.parameters(),  'lr': lr_actor},
            {'params': self.policy.skip_encoder.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor.parameters(),        'lr': lr_actor},
            {'params': [self.policy.log_std],                 'lr': lr_actor},
            {'params': self.policy.critic.parameters(),       'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Safeguard (APN)
        self.safeguard = ActionProjectionNetwork(
            state_dim=state_dim, action_dim=action_dim).to(device)
        _ckpt_paths = [
            os.path.join("policy", "action_projection_network.pth"),
            "action_projection_network.pth",
        ]
        for _ckpt in _ckpt_paths:
            if os.path.exists(_ckpt):
                self.safeguard.load_state_dict(
                    torch.load(_ckpt, map_location=device, weights_only=True))
                self.safeguard.eval()
                for p in self.safeguard.parameters():
                    p.requires_grad_(False)
                print(f"[Safeguard] Loaded APN from '{_ckpt}'.")
                break
        else:
            print("[Safeguard] WARNING: APN not found. Actions will be unfiltered.")

        self._proj_calls = 0
        self._proj_noop  = 0
        self._proj_iters = 0

        self.MseLoss           = nn.MSELoss()
        self.mapping_criterion = nn.MSELoss()

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

    def _project_to_safe(self, state_norm, action, max_steps=5, lr=1.0,
                          threshold=0.95):
        """
        Gradient-ascend the APN margin surface until classify() >= threshold.

        Stage-masked dimensions are zeroed in the gradient to prevent the
        projection from modifying locked action channels.

        Uses constant step size with mild decay to avoid premature stalling.
        """
        from env_core import PhycocyaninEnvCore

        a = action.clone().detach()
        state_fixed = state_norm.detach()
        self._proj_calls += 1

        # Stage mask for gradient zeroing
        stage_mask = PhycocyaninEnvCore.get_action_mask(state_fixed)

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
                    step_size = lr / (1.0 + step * 0.1)
                    a = a + step_size * grad.sign()
                    a = a.clamp(-1.0, 1.0)

        return best_a

    def select_action(self, state_norm):
        """
        Full safe action selection: Actor → mask → APN projection → final mask.
        """
        from env_core import PhycocyaninEnvCore

        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm).to(device).unsqueeze(0)

            # Generate intent via GRU actor (mask baked into act())
            z, log_prob, z_raw, self._hidden = self.policy_old.act(state_t, self._hidden)

            # Apply SERL mask checkpoint 2 (pre-APN)
            mask = PhycocyaninEnvCore.get_action_mask(state_t)
            default_squashed = torch.tensor([-0.333, -1.0, -1.0, -1.0], device=device)
            z_masked = z * mask + default_squashed * (1 - mask)

        # APN gradient projection
        u_safe = self._project_to_safe(state_t, z_masked)

        # Final SERL mask checkpoint 3 (post-APN)
        with torch.no_grad():
            u_safe = u_safe * mask + default_squashed * (1 - mask)

        return (u_safe.cpu().numpy().flatten(),
                log_prob.cpu().numpy(),
                z_raw.cpu().numpy().flatten())

    def learn(self, memory):
        """
        PPO update with margin-weighted mapping penalty.
        """
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states   = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_z_raw    = torch.squeeze(torch.stack(memory.raw_actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_z_raw)

            ratios     = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach().squeeze()
            surr1      = ratios * advantages
            surr2      = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            ppo_loss   = -torch.min(surr1, surr2).mean()

            # Margin-based mapping penalty
            with torch.no_grad():
                z_intent = torch.tanh(old_z_raw)
                margin   = self.safeguard(old_states, z_intent)
            mapping_penalty = torch.clamp(-margin, min=0.0).mean()

            loss = (ppo_loss
                    + 0.5  * self.MseLoss(state_values.squeeze(), rewards)
                    - self.entropy_coeff * dist_entropy.mean()
                    + 0.001 * mapping_penalty)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
