"""Action Projection Network (APN): the safe agent's L1 safety model and L2 projection.

Given a (state, action) pair the APN predicts one signed margin logit per
constraint (positive = safe):

  G1 — Nitrate path constraint  (cN + Fn*10h ≤ 800 mg/L)
  G2 — Product/biomass ratio    (cq/cx ≤ 0.011)
  G3 — Harvest-start nitrate    (cN ≤ 150 mg/L when Growth ends, still reachable
                                 after this action; learned from offline
                                 rollouts, safety/data.g3_margin())
  G4 — Reactor overflow         (V ≤ V_MAX)

G3 is a constraint at the end of a stage, but its head is one forward pass like
the others: the forecasting happened when the dataset was built.  G5 (end in
Idle) is left to the policy and the Lagrangian multipliers.  The APN covers
the stages in which the agent drives the culture (Inoculation, Growth); in
Harvesting and Idle it is bypassed.

Architecture (FiLM-conditioned, spectral-normalized):
  State encoder: Linear(12→256) → LN → Mish → Linear(256→256) → LN → Mish
  FiLM generators: 3× Linear(256→512) producing (γ,β) pairs
  Action trunk: Linear(4→256) → 3× [FC → LN → FiLM → Mish] (spectral-normed)
  → Linear(256→1) [raw margin]
  + 4× per-constraint heads (g1, g2, g3, g4)
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Local ────────────────────────────────────────────────────────────────────
from envs.plant import PhycocyaninEnvCore
from safety.constants import (APN_PATH, G3_FALLBACK, LR_PROJ, MAX_PROJ_STEPS, SAFE_LOGIT,
                              THRESHOLD)

# Temperature of the smooth minimum over the heads that the projection ascends.
SMOOTH_TAU = 0.25


def projection_active(state_norm):
    """True where the APN applies (Inoculation or Growth), shape [...]."""
    return (state_norm[..., 6] < 0.5) & (state_norm[..., 7] < 0.5)


class ActionProjectionNetwork(nn.Module):
    """Safety Proximity Regressor for the multi-stage photobioreactor.

    Given a (state, action) pair, outputs a continuous signed margin score:
      - margin > 0: Action is safe.
      - margin < 0: Action is unsafe.
      - margin = 0: Action is precisely on the safety boundary.

    Architecture (FiLM-conditioned with spectral normalization):
        State encoder: Linear(12→256) → Mish → Linear(256→256) → Mish
        State → FiLM params: 3× (γ, β) pairs for action trunk modulation
        Action trunk: Linear(4→256) → [FiLM block]×3 → Linear(256→1)
        Spectral norm on action-path layers for Lipschitz-bounded ∂margin/∂action.
        Also includes 4 per-constraint heads (g1, g2, g3, g4), which the
        projection and the safety gradient use.

    Attributes:
        use_g3: Include the G3 head in margins() (ablation switch).
        has_g3: The G3 head is trained (False for a checkpoint saved before it
            existed, whose g3_head would be random; margins() then leaves it out).
    """

    def __init__(self, state_dim: int = 12, action_dim: int = 4, latent_dim: int = 256):
        """Initializes the Action Projection Network with FiLM conditioning.

        The state is encoded separately and produces scale/shift (γ, β) parameters
        that modulate the action processing trunk. This yields cleaner ∂margin/∂action
        gradients for the projection step. Spectral normalization ensures Lipschitz
        continuity for stable gradient ascent convergence.

        Args:
            state_dim (int, optional): Dimension of observation space. Defaults to 12.
            action_dim (int, optional): Dimension of action space. Defaults to 4.
            latent_dim (int, optional): Width of hidden layers. Defaults to 256.
        """
        super(ActionProjectionNetwork, self).__init__()
        self.latent_dim = latent_dim

        # ── State encoder (no spectral norm — not differentiated during projection)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
        )

        # FiLM generators: state → (γ, β) for each of 3 action trunk blocks.
        # Zero-initialised so all generators start as identity (γ=0, β=0),
        # preventing activation explosions before the first gradient update.
        self.film1 = nn.Linear(latent_dim, latent_dim * 2)
        self.film2 = nn.Linear(latent_dim, latent_dim * 2)
        self.film3 = nn.Linear(latent_dim, latent_dim * 2)
        for _film in (self.film1, self.film2, self.film3):
            nn.init.zeros_(_film.weight)
            nn.init.zeros_(_film.bias)

        # ── Action trunk (spectral-normed for Lipschitz-bounded action gradients)
        self.action_lift = nn.utils.spectral_norm(nn.Linear(action_dim, latent_dim))

        self.trunk_fc1 = nn.utils.spectral_norm(nn.Linear(latent_dim, latent_dim))
        self.trunk_ln1 = nn.LayerNorm(latent_dim)
        self.trunk_fc2 = nn.utils.spectral_norm(nn.Linear(latent_dim, latent_dim))
        self.trunk_ln2 = nn.LayerNorm(latent_dim)
        self.trunk_fc3 = nn.utils.spectral_norm(nn.Linear(latent_dim, latent_dim))
        self.trunk_ln3 = nn.LayerNorm(latent_dim)

        self.act = nn.Mish()

        # ── Output heads
        # spectral_norm wraps weight_orig — do NOT zero-init weight_orig as
        # σ = u^T W v → 0 causes NaN (division by zero in the SN forward hook).
        self.margin_head = nn.utils.spectral_norm(nn.Linear(latent_dim, 1))
        nn.init.zeros_(self.margin_head.bias)

        # Per-constraint heads (direct gradient signal per constraint)
        self.g1_head = nn.Linear(latent_dim, 1)
        self.g2_head = nn.Linear(latent_dim, 1)
        self.g3_head = nn.Linear(latent_dim, 1)
        self.g4_head = nn.Linear(latent_dim, 1)

        self.use_g3 = True
        self.has_g3 = True

    def _encode_state(self, state_norm):
        """Encodes the state into FiLM conditioning parameter tensors.

        Args:
            state_norm (torch.Tensor): The normalized observation vector.

        Returns:
            tuple: A tuple containing:
                - h_s (torch.Tensor): The state encoder embedding tensor.
                - film1 (torch.Tensor): Conditioning parameter tensor for block 1.
                - film2 (torch.Tensor): Conditioning parameter tensor for block 2.
                - film3 (torch.Tensor): Conditioning parameter tensor for block 3.
        """
        h_s = self.state_encoder(state_norm)
        film1 = self.film1(h_s)
        film2 = self.film2(h_s)
        film3 = self.film3(h_s)
        return h_s, film1, film2, film3

    def _apply_film(self, x, film_params):
        """Applies FiLM modulation: x * (1 + gamma) + beta.

        Args:
            x (torch.Tensor): Input activation tensor.
            film_params (torch.Tensor): Split conditioning parameters.

        Returns:
            torch.Tensor: The modulated activation tensor.
        """
        gamma, beta = film_params.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta

    def _action_trunk(self, action_norm, film1, film2, film3):
        """Processes action tensor through the FiLM-modulated trunk.

        Args:
            action_norm (torch.Tensor): Normalized action vector.
            film1 (torch.Tensor): Modulation parameters for the first layer.
            film2 (torch.Tensor): Modulation parameters for the second layer.
            film3 (torch.Tensor): Modulation parameters for the third layer.

        Returns:
            torch.Tensor: The processed action representation tensor.
        """
        h = self.act(self.action_lift(action_norm))

        h = self.trunk_fc1(h)
        h = self.trunk_ln1(h)
        h = self._apply_film(h, film1)
        h = self.act(h)

        h = self.trunk_fc2(h)
        h = self.trunk_ln2(h)
        h = self._apply_film(h, film2)
        h = self.act(h)

        h = self.trunk_fc3(h)
        h = self.trunk_ln3(h)
        h = self._apply_film(h, film3)
        h = self.act(h)

        return h

    def forward(self, state_norm, action_norm):
        """Computes the continuous signed safety margin for the given state and action.

        Args:
            state_norm (torch.Tensor): Normalized observation vector.
            action_norm (torch.Tensor): Action vector.

        Returns:
            torch.Tensor: Continuous margin score.
        """
        _, film1, film2, film3 = self._encode_state(state_norm)
        h = self._action_trunk(action_norm, film1, film2, film3)
        margin = self.margin_head(h).squeeze(-1)
        return margin

    def forward_aux(self, state_norm, action_norm):
        """Forward pass returning main margin and per-constraint auxiliary margins.

        Args:
            state_norm (torch.Tensor): Normalized observation vector.
            action_norm (torch.Tensor): Action vector.

        Returns:
            tuple: (margin, g1_margin, g2_margin, g3_margin, g4_margin)
        """
        _, film1, film2, film3 = self._encode_state(state_norm)
        h = self._action_trunk(action_norm, film1, film2, film3)
        margin = self.margin_head(h).squeeze(-1)
        g1 = self.g1_head(h).squeeze(-1)
        g2 = self.g2_head(h).squeeze(-1)
        g3 = self.g3_head(h).squeeze(-1)
        g4 = self.g4_head(h).squeeze(-1)
        return margin, g1, g2, g3, g4

    def classify(self, state_norm, action_norm):
        """Returns the probability that the state-action pair is safe (margin > 0).

        Args:
            state_norm (torch.Tensor): Normalized observation vector.
            action_norm (torch.Tensor): Action vector.

        Returns:
            torch.Tensor: Safety probability [0, 1].
        """
        return torch.sigmoid(self.forward(state_norm, action_norm))

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device, **kwargs) -> "ActionProjectionNetwork":
        """Instantiate an APN and load a checkpoint with shape-safe partial loading.

        Keys present in the checkpoint but with a shape mismatch against the
        current architecture are silently skipped, so weights always load without
        crashing across architecture changes (e.g. latent_dim 160 → 256).

        Args:
            path (str): Path to the .pth checkpoint file.
            device (torch.device): Target device.
            **kwargs: Forwarded to __init__ (e.g. state_dim, action_dim, latent_dim).

        Returns:
            ActionProjectionNetwork: Model with compatible weights loaded.
        """
        model = cls(**kwargs).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        current = model.state_dict()
        compatible = {k: v for k, v in ckpt.items()
                      if k in current and current[k].shape == v.shape}
        skipped = len(ckpt) - len(compatible)
        model.load_state_dict(compatible, strict=False)
        model.has_g3 = "g3_head.weight" in compatible
        print(f"[APN] Loaded '{path}' — {len(compatible)} tensors restored, "
              f"{skipped} skipped (shape mismatch / new layers)."
              + ("" if model.has_g3 else " No trained G3 head: G3 is not covered "
                 "(retrain with python -m safety.train)."))
        return model

    # ── L1: Per-constraint margins ────────────────────────────────────────────

    def margins(self, state_norm, action_norm):
        """Per-constraint margin logits [..., H] in (g1, g2, g3, g4) order.

        G3 is left out (H = 3) when it is switched off or untrained.
        """
        _, g1, g2, g3, g4 = self.forward_aux(state_norm, action_norm)
        heads = [g1, g2, g3, g4] if (self.use_g3 and self.has_g3) else [g1, g2, g4]
        return torch.stack(heads, dim=-1)

    def _smooth_min(self, state_norm, action_norm):
        """Smooth minimum of the per-constraint margins (the projection objective)."""
        m = self.margins(state_norm, action_norm)
        return -SMOOTH_TAU * torch.logsumexp(-m / SMOOTH_TAU, dim=-1)

    # ── L2: Projection ────────────────────────────────────────────────────────

    def project(self, state_norm, action, lr=LR_PROJ, max_steps=MAX_PROJ_STEPS,
                threshold=THRESHOLD):
        """Gradient-ascend the margin surface to find a safe action proxy.

        If the proposed action is unsafe, ascends the smooth minimum of the
        per-constraint margins in action space with sign steps.  Stage-masked
        dimensions are zeroed in the gradient so locked channels never move.
        Harvesting and Idle states pass through unchanged.

        Args:
            state_norm: Observation tensor [1, state_dim].
            action: Squashed, stage-masked action [1, action_dim].
            lr: Sign-step size (mildly decayed per step).
            max_steps: Maximum ascent steps.
            threshold: Safety probability at which an action is accepted.

        Returns:
            A tuple of the projected action [1, action_dim] (the best iterate
            if the threshold is not reached) and a stats dict with 'noop'
            (bool) and 'iters' (ascent steps taken).
        """
        a = action.clone().detach()
        state_fixed = state_norm.detach()
        stats = {"noop": True, "iters": 0}

        # Bypass projection in Stage 2 (Harvesting) and Stage 3 (Idle) to avoid
        # phantom gradients lock-in during draining and shutdown.
        if not bool(projection_active(state_fixed).all()):
            return a, stats

        with torch.no_grad():
            p = torch.sigmoid(self.margins(state_fixed, a)).min(dim=-1).values
            if p.item() >= threshold:
                return a, stats
            best_margin = self._smooth_min(state_fixed, a).item()
        stats["noop"] = False

        stage_mask = PhycocyaninEnvCore.get_action_mask(state_fixed)
        best_a = a.clone()
        with torch.enable_grad():
            for step in range(max_steps):
                a_var = a.clone().requires_grad_(True)
                margin = self._smooth_min(state_fixed, a_var)

                if torch.sigmoid(margin).item() >= threshold:
                    return a_var.detach(), stats

                # Track best iterate seen
                m_val = margin.item()
                if m_val > best_margin:
                    best_margin = m_val
                    best_a = a_var.detach().clone()

                grad = torch.autograd.grad(margin.sum(), a_var)[0]
                stats["iters"] += 1

                with torch.no_grad():
                    # Zero gradient for masked dimensions
                    grad = grad * stage_mask

                    at_lower = (a_var <= -1.01) & (grad < 0)
                    at_upper = (a_var >= 1.01) & (grad > 0)
                    grad[at_lower | at_upper] = 0.0

                    # Constant step size with mild decay
                    step_size = lr / (1.0 + step * 0.03)

                    # Boost step size for Fn (index 2) when volume is high to quickly avoid overflow
                    lr_mult = torch.ones_like(grad)
                    lr_mult[..., 2] = torch.where(state_fixed[..., 3] > 0.75, 5.0, 1.0)

                    a = (a + step_size * grad.sign() * lr_mult).clamp(-1.0, 1.0)

        if self.use_g3 and self.has_g3:
            best_a = self._toward_fallback(state_fixed, best_a, stage_mask, threshold)
            stats["iters"] += 1
        return best_a, stats

    def _toward_fallback(self, state_norm, action, stage_mask, threshold, n=4):
        """Line search from an action toward the G3 fallback, in one batched forward pass.

        The fallback (no feed, slowest clock, growth-optimal light) is the most
        G3-safe action there is, so when sign steps run out on the way there
        the closest of n evenly spaced points on the segment that every head
        accepts is taken (the fallback itself, the safest, if none is).

        Args:
            state_norm: Observation tensor [1, state_dim].
            action: Action [1, action_dim] the sign steps ended at.
            stage_mask: Stage action mask [1, action_dim]; locked dims stay put.
            threshold: Safety probability at which an action is accepted.

        Returns:
            The chosen action [1, action_dim].
        """
        fallback = torch.tensor(G3_FALLBACK, device=action.device, dtype=action.dtype)
        target = fallback * stage_mask + action * (1 - stage_mask)
        lam = torch.linspace(1.0 / n, 1.0, n, device=action.device, dtype=action.dtype).unsqueeze(1)
        cand = action + lam * (target - action)                                # [n, action_dim]
        with torch.no_grad():
            p = torch.sigmoid(self.margins(state_norm.expand(n, -1), cand)).min(dim=-1).values
        ok = (p >= threshold).nonzero(as_tuple=True)[0]
        k = int(ok[0]) if ok.numel() else n - 1
        return cand[k:k + 1]

    # ── Safety Gradient ───────────────────────────────────────────────────────

    def safety_loss(self, state_norm, action, sharpness=5.0, eps=SAFE_LOGIT):
        """Differentiable penalty on every head's shortfall (the actor's safety gradient).

        A smooth hinge, softplus(sharpness * (eps - m)) / sharpness, per head:
        actions above the projection's acceptance logit contribute almost
        nothing and each violated head pulls the action along its own margin
        gradient.  Harvesting and Idle rows are excluded.  Use with frozen APN
        weights: the gradient reaches the actor only through ``action``.

        Args:
            state_norm: Tensor [..., state_dim].
            action: Tensor [..., action_dim], a differentiable function of the actor.
            sharpness: Hinge sharpness (the margins are logits).
            eps: Margin each head is pushed up to.

        Returns:
            Scalar loss (mean over the active rows' heads).
        """
        m = self.margins(state_norm, action)
        hinge = F.softplus(sharpness * (eps - m)) / sharpness          # [..., H]
        active = projection_active(state_norm).to(hinge.dtype).unsqueeze(-1)
        return (hinge * active).sum() / (active.sum() * m.shape[-1]).clamp(min=1.0)


def load_apn(device, state_dim=12, action_dim=4):
    """The trained APN, frozen and in eval mode, or None if no checkpoint exists."""
    for ckpt in (APN_PATH, "action_projection_network.pth"):
        try:
            apn = ActionProjectionNetwork.from_checkpoint(
                ckpt, device, state_dim=state_dim, action_dim=action_dim)
        except FileNotFoundError:
            continue
        apn.eval()
        for p in apn.parameters():
            p.requires_grad_(False)
        return apn
    return None
