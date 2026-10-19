"""Recurrent actor-critic of the Safe PPO agent and the backend it runs on.

    ActorCritic   GRU actor-critic (DRPO) with a skip connection from the raw state.
    device        Torch device every agent component uses.

The photobioreactor is a single reactor, so the Fermentation project's
cross-reactor attention has no counterpart here.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.distributions import Normal

# ─── Local ────────────────────────────────────────────────────────────────────
from envs.plant import DEFAULT_INTENT, PhycocyaninEnvCore

# ─── Recurrent Policy Hyperparameters ─────────────────────────────────────────

# Width of the GRU hidden state.
GRU_HIDDEN_DIM = 128

# Number of stacked GRU layers.
GRU_NUM_LAYERS = 1

# Sequence length for Truncated Backpropagation Through Time (TBPTT) during
# the PPO update.  Gradients flow through exactly SEQ_LEN GRU steps per chunk.
# 32 control steps = 320 h, about one Inoculation + Growth batch.
SEQ_LEN = 32

# ─── Backend Configuration ────────────────────────────────────────────────────

# Disable CuDNN and MIOpen backends to avoid ROCm GRU bugs (miopenStatusUnknownError)
torch.backends.cudnn.enabled = False
if hasattr(torch.backends, 'miopen'):
    torch.backends.miopen.enabled = False

# Detect hardware acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Actor-Critic Architecture ────────────────────────────────────────────────


class ActorCritic(nn.Module):
    """Recurrent Actor-Critic with a raw-state skip connection (DRPO).

    The GRU provides temporal memory so the agent can anticipate constraint
    violations (especially g2 ratio buildup and g3 terminal nitrate) before
    they occur.  A skip connection concatenates the GRU output with the raw
    state for both heads.

    Architecture::

        gru        GRU(state_dim -> 128, batch_first=True)
        actor      MLP: 128 + state_dim -> 256 -> 256 -> action_dim   (ELU)
        log_std    nn.Parameter(action_dim)
        critic     MLP: 128 + state_dim -> 256 -> 256 -> 1            (ELU)

    Action masking (SERL L0) is baked into the distribution:
        - Masked dims get a fixed default intent (pre-Tanh)
        - Masked dims get near-zero std to suppress exploration
        - Log-probs and entropy exclude masked dims

    Attributes:
        LOG_STD_MIN: Lower bound for log_std clamping.
        LOG_STD_MAX: Upper bound for log_std clamping.
        env_state_dim: Raw environment state width (for mask extraction).
        gru_hidden: GRU hidden state width.
    """

    def __init__(self, state_dim=12, action_dim=4, use_safeguard=True):
        """Initialise the recurrent actor-critic network.

        Args:
            state_dim: Observation dimensionality.
            action_dim: Number of action dimensions.
            use_safeguard: If True, applies the SERL stage mask to mean/std.
        """
        super().__init__()
        self.LOG_STD_MIN = -1.0
        self.LOG_STD_MAX = 0.5
        self.env_state_dim = state_dim
        self.gru_hidden = GRU_HIDDEN_DIM
        self.use_safeguard = use_safeguard

        # ── Shared temporal memory ─────────────────────────────────────────────
        self.gru = nn.GRU(state_dim, GRU_HIDDEN_DIM, num_layers=GRU_NUM_LAYERS,
                          batch_first=True)

        # ── Actor and critic heads (GRU features + raw-state skip) ─────────────
        head_in = GRU_HIDDEN_DIM + state_dim
        self.actor = nn.Sequential(
            nn.Linear(head_in, 256), nn.ELU(),
            nn.Linear(256, 256),     nn.ELU(),
            nn.Linear(256, action_dim)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        self.critic = nn.Sequential(
            nn.Linear(head_in, 256), nn.ELU(),
            nn.Linear(256, 256),     nn.ELU(),
            nn.Linear(256, 1)
        )

    def init_hidden(self, batch_size=1):
        """Zero hidden state of shape [num_layers, batch_size, gru_hidden]."""
        return torch.zeros(GRU_NUM_LAYERS, batch_size, self.gru_hidden,
                           device=self.log_std.device)

    def _encode_sequence(self, state_seq, h_init, done_seq=None):
        """Encode an observation sequence through the GRU and skip paths.

        Args:
            state_seq: Observation tensor of shape [B, T, state_dim].
            h_init: Initial hidden state of shape [num_layers, B, gru_hidden].
            done_seq: Optional float tensor of shape [B, T] with 1.0 where an
                episode ends; the hidden state is reset to zeros after it.

        Returns:
            A tuple of:
                - feat: Concatenated features of shape [B, T, gru_hidden + state_dim].
                - h: Final GRU hidden state of shape [num_layers, B, gru_hidden].
        """
        B, T, _ = state_seq.shape
        # A done at the last step only matters to the next sequence.
        inner_done = None if done_seq is None else (done_seq[:, :-1] > 0.5)

        if inner_done is None or not bool(inner_done.any()):
            # Fast path when no episode boundaries need explicit handling.
            gru_out, h = self.gru(state_seq, h_init)
        elif B == 1:
            # Segment-based processing: run the GRU on each contiguous episode
            # segment in a single call rather than T single-step calls.
            h = h_init
            outs = []
            boundary_steps = (inner_done[0].nonzero(as_tuple=True)[0] + 1).tolist()
            boundary_steps.append(T)   # sentinel: always process to end
            seg_start = 0
            for seg_end in boundary_steps:
                out_seg, h = self.gru(state_seq[:, seg_start:seg_end], h)
                outs.append(out_seg)
                seg_start = seg_end
                if seg_start < T:
                    h = torch.zeros_like(h)   # reset at episode boundary
            gru_out = torch.cat(outs, dim=1)
        else:
            # Rows end their episodes at different steps: step the GRU and
            # zero each row's hidden state after its own boundaries.
            keep = 1.0 - done_seq.to(state_seq.dtype)            # [B, T]
            h = h_init
            outs = []
            for t in range(T):
                out_t, h = self.gru(state_seq[:, t:t + 1], h)
                outs.append(out_t)
                h = h * keep[:, t].view(1, B, 1)
            gru_out = torch.cat(outs, dim=1)

        feat = torch.cat([gru_out, state_seq], dim=-1)
        return torch.nan_to_num(feat, nan=0.0), h

    def _distribution(self, feat, state):
        """Stage-masked Gaussian over the pre-tanh intent.

        Returns:
            A tuple of (dist, mask, masked_mean).
        """
        # float32 even under AMP: the masked dims' 1e-8 std underflows in fp16.
        mean = torch.nan_to_num(self.actor(feat).float(), nan=0.0)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))

        if self.use_safeguard:
            mask = PhycocyaninEnvCore.get_action_mask(state[..., :self.env_state_dim]).to(mean.dtype)
            default_intent = torch.tensor(DEFAULT_INTENT, device=mean.device, dtype=mean.dtype)
            masked_mean = mean * mask + default_intent * (1 - mask)
            masked_std = std * mask + 1e-8 * (1 - mask)
        else:
            mask = torch.ones_like(mean)
            masked_mean = mean
            masked_std = std.expand_as(mean)
        return Normal(masked_mean, masked_std), mask, masked_mean

    def act(self, state, h=None):
        """Sample a latent intent for the current control step and advance h.

        Args:
            state: Current observation tensor of shape [B, state_dim].
            h: Recurrent hidden state [num_layers, B, gru_hidden], or None at
                the start of an episode.

        Returns:
            A tuple of:
                - z: Sampled raw intent of shape [B, action_dim] (pre-tanh).
                - log_prob: Log-probability log pi(z | s, h) of shape [B].
                - h_new: Updated hidden state.
        """
        if h is None:
            h = self.init_hidden(state.size(0))

        feat, h_new = self._encode_sequence(state.unsqueeze(1), h)
        feat = feat.squeeze(1)

        dist, mask, _ = self._distribution(feat, state)
        z = dist.sample()
        log_prob = (dist.log_prob(z) * mask).sum(dim=-1)

        return z.detach(), log_prob.detach(), h_new.detach()

    def evaluate(self, state_seq, z_seq, h_init, done_seq):
        """Evaluate a rollout chunk under the current (updated) policy.

        Gradients flow back through the GRU for the full window (TBPTT).

        Args:
            state_seq: Batch-first observation chunk of shape [B, T, state_dim].
            z_seq: Stored raw intents of shape [B, T, action_dim].
            h_init: Chunk initial hidden state [num_layers, B, gru_hidden].
            done_seq: Float tensor of shape [B, T] with 1.0 at episode ends.

        Returns:
            A tuple of:
                - log_probs: Log-probabilities of shape [B, T].
                - state_values: Value estimates of shape [B, T, 1].
                - dist_entropy: Entropy of shape [B, T].
                - mean: The (stage-masked) actor mean [B, T, action_dim],
                  differentiable, for the safety gradient.
        """
        feat, _ = self._encode_sequence(state_seq, h_init, done_seq)

        dist, mask, masked_mean = self._distribution(feat, state_seq)
        log_probs = (dist.log_prob(z_seq) * mask).sum(dim=-1)
        dist_entropy = (dist.entropy() * mask).sum(dim=-1)
        state_values = self.critic(feat)

        return log_probs, state_values, dist_entropy, masked_mean
