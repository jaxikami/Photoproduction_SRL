"""Multi-head Action Projection Network (APN): the safe agent's safety layer.

Laid out like Fermentation-PPO's MultiHeadAPN.  The APN scores a (state,
action) pair with one signed margin per constraint head (positive: safe by
that much; negative: violated by that much, both as a fraction of the limit)
and moves an unsafe action back inside every head's boundary.  It works in
two stages:

    L1  estimate   The margin of every head.  Exact heads are the plant's own
                   one-step kinetics in torch (safety/dynamics.py); the
                   learned head is a network (KineticNet) trained on the
                   fallback's nitrate draw-down (safety/train.py).
    L2  correct    A closed-form minimum-norm step on the heads that are short
                   of their boundary, damped by priority so the more important
                   head wins a conflict, then a final pass on the containment
                   heads alone.  No optimizer and no iterations beyond a few
                   re-linearizations; a row G3 is still short on after them
                   takes the closest accepted point toward the G3 fallback.

Every margin is differentiable in the action, so the same heads give the actor
a safety gradient during the PPO update (safety_loss()).

Heads (constraint id in brackets):

    g1  [1] exact    CN after the control step <= SAFE_BUFFER x 800 mg/L
    g2  [2] exact    Cq/Cx after the control step <= SAFE_BUFFER x 0.011
    g3  [3] learned  CN when Growth ends <= G3_BUFFER x 150 mg/L, if the
                     G3 fallback (no feed, slowest clock, growth-optimal
                     light) follows this action: still reachable after it
    g4  [4] exact    V after the control step <= SAFE_BUFFER x V_MAX

The G3 head splits the forecast like Fermentation's heat head: KineticNet
predicts the fallback's nitrate at the end of each coming control step from
the biomass and nitrate the exact one-step kinetics give after the action
(neither volume, product nor the clock enters that draw-down), and which step
ends Growth is the plant's credit arithmetic.  A stage ends on the step its
credits run out, so the step index jumps with the clock; it is computed
exactly, with the gradient of its linear interpolation.  The APN covers the
stages in which the agent drives the culture (Inoculation, Growth); in
Harvesting and Idle it is bypassed.  G5 (end in Idle) is left to the policy
and the Lagrangian multipliers.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import math
import os
from dataclasses import dataclass

# ─── Third-Party ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Local ────────────────────────────────────────────────────────────────────
from envs.plant import PhycocyaninEnvCore
from safety.constants import (
    BASE_CREDITS, CONTAINMENT_ITERS, CONTROL_INTERVAL, DRAWDOWN_STEPS, FALLBACK_CREDITS,
    FN_MAX_GROWTH, FN_MAX_PROD, G3_BUFFER, G3_EPS, G3_FALLBACK, I_MAX, I_MIN, KINETIC_PATH, MULT_MIN, MULT_RANGE,
    N_LIMIT_PATH, N_LIMIT_TERM, PROJ_DAMPING, PROJ_EPS, PROJ_ITERS, RATIO_LIMIT, SAFE_BUFFER,
    TOTAL_TIME, V_MAX
)
from safety.dynamics import control_step
from safety.kinetic import KineticNet

ACTION_DIM = 4
MAX_STEPS = int(round(TOTAL_TIME / CONTROL_INTERVAL))
# Fallback steps a fresh Growth lasts.
_GROWTH_STEPS = float(math.ceil(BASE_CREDITS[1] / FALLBACK_CREDITS))

# ─── Head Registry ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HeadSpec:
    """One constraint head.

    Attributes:
        name: Head name (key of head_margins()).
        group: Violation id the head guards (key of its buffer offset).
        exact: Computed from the plant's own kinetics (else KineticNet).
        priority: Weight in the correction step: the higher one wins a conflict.
        dims: Action dims the head may move (0 clock, 1 light, 2 feed, 3 outflow).
    """
    name: str
    group: int
    exact: bool
    priority: float
    dims: tuple


HEADS = (
    HeadSpec("g1", 1, True, 100.0, (1, 2)),
    HeadSpec("g2", 2, True, 30.0, (1, 2)),
    HeadSpec("g3", 3, False, 30.0, (0, 1, 2)),
    HeadSpec("g4", 4, True, 100.0, (2,)),
)
# Physical containment: re-checked alone after the joint correction so a
# compromise with a softer head can never leave them violated.
CONTAINMENT_GROUPS = (1, 4)


def projection_active(state_norm):
    """True where the APN applies (Inoculation or Growth), shape [...]."""
    return (state_norm[..., 6] < 0.5) & (state_norm[..., 7] < 0.5)


# ─── L1 + L2: the APN ─────────────────────────────────────────────────────────


class MultiHeadAPN(nn.Module):
    """Margins of every constraint head, and the projection built on them.

    Attributes:
        kinetic: KineticNet behind the G3 head.
        use_g3: Include the G3 head (ablation switch).
        has_g3: KineticNet is trained (False without a checkpoint, in which
            case the G3 head is left out).
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.kinetic = KineticNet(hidden)
        self.use_g3 = True
        self.has_g3 = False

    def load_kinetic(self, path=KINETIC_PATH, map_location=None):
        """Load trained KineticNet weights; returns False when there are none."""
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        self.kinetic.load_state_dict(ckpt["state_dict"])
        self.has_g3 = True
        return True

    @property
    def heads(self):
        """The active HeadSpecs, in margin column order."""
        return tuple(h for h in HEADS if h.exact or (self.use_g3 and self.has_g3))

    # ── Margins ───────────────────────────────────────────────────────────────

    def head_margins(self, state, action, offsets=None):
        """Signed margin of every head.

        Args:
            state: Tensor [N, 12] (normalised observation).
            action: Tensor [N, 4] in [-1, 1], stage-masked.
            offsets: Optional dict group -> buffer offset (fraction of the
                limit) taken off that group's buffered limit.

        Returns:
            Dict head name -> margin [N]; +1 where the APN does not apply.
        """
        off = offsets or {}
        state, action = state.float(), action.float()
        stage = state[:, 4:8].argmax(dim=1)
        growth = stage == 1
        u = (action.clamp(-1.0, 1.0) + 1.0) * 0.5
        mult = MULT_MIN + u[:, 0] * MULT_RANGE
        light = I_MIN + u[:, 1] * (I_MAX - I_MIN)
        fn_max = torch.where(growth, torch.full_like(mult, FN_MAX_PROD),
                             torch.full_like(mult, FN_MAX_GROWTH))
        feed = u[:, 2] * fn_max * (state[:, 10] > 0.0).float()

        x = torch.stack([state[:, 0] * 6.0, state[:, 1] * 800.0, state[:, 2] * 0.2,
                         state[:, 3] * V_MAX], dim=-1)
        x1 = control_step(x, light, feed, torch.zeros_like(feed))
        cx1, cn1, cq1, v1 = x1.unbind(-1)

        out = {
            "g1": ((SAFE_BUFFER - off.get(1, 0.0)) * N_LIMIT_PATH - cn1) / N_LIMIT_PATH,
            "g2": ((SAFE_BUFFER - off.get(2, 0.0)) * RATIO_LIMIT - cq1 / (cx1 + 1e-8)) / RATIO_LIMIT,
            "g4": ((SAFE_BUFFER - off.get(4, 0.0)) * V_MAX - v1) / V_MAX,
        }
        if self.use_g3 and self.has_g3:
            out["g3"] = self._g3_margin(state, stage, mult, cx1, cn1, off.get(3, 0.0))

        active = projection_active(state)
        return {k: torch.where(active, m, torch.ones_like(m)) for k, m in out.items()}

    def _g3_margin(self, state, stage, mult, cx1, cn1, offset):
        """G3 margin: nitrate when Growth ends if the fallback follows the action.

        Returns:
            Margin [N] (+1 where the episode ends before Growth does).
        """
        base = torch.where(stage == 1, torch.full_like(mult, BASE_CREDITS[1]),
                           torch.full_like(mult, BASE_CREDITS[0]))
        left = state[:, 8] * base - mult * CONTROL_INTERVAL      # credits after the action
        # Fallback steps until Growth ends: the rest of the current stage
        # (a stage ends on the step its credits reach 0), plus a whole Growth
        # after Inoculation.
        rest = left / FALLBACK_CREDITS
        extra = torch.where(stage == 0, torch.full_like(rest, _GROWTH_STEPS), torch.zeros_like(rest))
        k_exact = (torch.ceil(rest - 1e-6).clamp(min=0.0) + extra).clamp(max=DRAWDOWN_STEPS)
        # Unclamped below: where this step already ends the stage, the
        # extrapolated profile still tells the clock that a slower one helps.
        k_soft = (rest + extra).clamp(max=DRAWDOWN_STEPS)

        # Nitrate after the action, then at the end of every fallback step.
        profile = torch.cat([cn1.unsqueeze(-1), self.kinetic(cx1, cn1)], dim=-1)
        exact = profile.gather(1, k_exact.long().unsqueeze(-1)).squeeze(-1)
        k0 = torch.floor(k_soft).clamp(0.0, DRAWDOWN_STEPS - 1).detach()
        w = k_soft - k0
        lo = profile.gather(1, k0.long().unsqueeze(-1)).squeeze(-1)
        hi = profile.gather(1, (k0 + 1).long().unsqueeze(-1)).squeeze(-1)
        interp = lo + w * (hi - lo)
        # The exact step's nitrate, with the interpolation's gradient (the
        # step index itself is piecewise constant in the clock).
        cn_end = interp + (exact - interp).detach()

        margin = ((G3_BUFFER - offset) * N_LIMIT_TERM - cn_end) / N_LIMIT_TERM
        steps_now = torch.round(state[:, 9] * MAX_STEPS)
        checked = steps_now + 1.0 + k_exact <= MAX_STEPS
        return torch.where(checked, margin, torch.ones_like(margin))

    def margins(self, state, action, offsets=None):
        """Margins [..., H] in heads order, for any leading shape."""
        lead = state.shape[:-1]
        heads = self.head_margins(state.reshape(-1, state.shape[-1]),
                                  action.reshape(-1, action.shape[-1]), offsets)
        return torch.stack([heads[h.name] for h in self.heads], dim=-1).reshape(*lead, -1)

    def margins_and_jacobian(self, state, action, offsets=None):
        """Margins and their Jacobian in the action.

        Rows are independent, so the batch is repeated once per head and copy
        k keeps only head k: one backward pass gives every head's rows.

        Returns:
            A tuple of margins [N, H] and Jacobian [N, H, 4] (detached).
        """
        n, h = action.shape[0], len(self.heads)
        with torch.enable_grad():
            a = action.detach().float().repeat(h, 1).requires_grad_(True)       # [h*n, 4]
            m = self.margins(state.repeat(h, 1), a, offsets).reshape(h, n, h)
            pick = torch.eye(h, device=a.device).unsqueeze(1)                   # [h, 1, h]
            g = torch.autograd.grad((m * pick).sum(), a)[0].reshape(h, n, ACTION_DIM)
        return m[0].detach(), g.transpose(0, 1).detach()

    # ── L2: Correction ────────────────────────────────────────────────────────

    @staticmethod
    def _correct(a, m, J, active, priority, eps, damping):
        """One minimum-norm step that brings every active margin up to eps.

        Solves  min ||delta||  s.t.  m_k + J_k delta = eps  (k active), with
        each equation softened by damping / priority: the k x k system is
        solved in closed form, so a lower-priority head yields in a conflict.
        Entries pinned at a bound and pushed further out are frozen.

        Args:
            a: Actions [N, 4].
            m: Margins [N, H].
            J: Jacobian [N, H, 4] restricted to the movable entries.
            active: Bool [N, H].
            priority: Tensor [H].
            eps: Target margin [H].
            damping: Base damping.

        Returns:
            Corrected actions [N, 4] clamped to [-1, 1].
        """
        pinned = (((a >= 1.0 - 1e-4).unsqueeze(1) & (J > 0))
                  | ((a <= -1.0 + 1e-4).unsqueeze(1) & (J < 0)))
        J = J.masked_fill(pinned, 0.0)
        # A head with no lever left cannot be helped by this step.
        act = (active & (J.pow(2).sum(-1) > 1e-10)).to(J.dtype)
        A = J * act.unsqueeze(-1)
        r = (eps - m).clamp(min=0.0) * act
        G = A @ A.transpose(1, 2) + torch.diag_embed(damping / priority + (1.0 - act))
        y = torch.linalg.solve(G, r.unsqueeze(-1))
        delta = (A.transpose(1, 2) @ y).squeeze(-1)
        return (a + delta).clamp(-1.0, 1.0)

    @torch.no_grad()
    def project(self, state, action, offsets=None, iters=PROJ_ITERS,
                containment_iters=CONTAINMENT_ITERS, eps=PROJ_EPS, damping=PROJ_DAMPING):
        """L2: move an action inside every head's boundary.

        Args:
            state: Tensor [N, 12].
            action: Tensor [N, 4] in [-1, 1], stage-masked.
            offsets: Buffer offsets (agents/safe.PPO_Agent._offsets()).
            iters: Joint correction steps (re-linearized each time).
            containment_iters: Final steps on the containment heads alone.
            eps: Margin each corrected exact head is brought up to (the
                learned G3 head: G3_EPS).
            damping: Base damping of the correction step.

        Returns:
            A tuple of the projected action [N, 4] and a stats dict with
            'noop' (bool), 'iters' (int) and 'groups' (set of constraint ids
            whose heads had to be corrected).
        """
        a = action.detach().float().clone()
        stats = {"noop": True, "iters": 0, "groups": set()}
        m = self.margins(state, a, offsets)
        if bool((m >= 0.0).all()):
            return a, stats
        stats["noop"] = False

        heads = self.heads
        dims = torch.zeros(len(heads), ACTION_DIM, device=a.device)
        for k, h in enumerate(heads):
            dims[k, list(h.dims)] = 1.0
        priority = torch.tensor([h.priority for h in heads], device=a.device)
        cont = torch.tensor([h.group in CONTAINMENT_GROUPS for h in heads], device=a.device)
        target = torch.tensor([eps if h.exact else G3_EPS for h in heads], device=a.device)
        move = dims.unsqueeze(0) * PhycocyaninEnvCore.get_action_mask(state.float()).unsqueeze(1)

        # Each pass: linearize only when some head is short, correct, then
        # re-check with a forward pass alone.
        for k in range(iters + containment_iters):
            if k == iters:
                a, m = self._toward_fallback(state, a, m, offsets)
            active = m < 0.0
            if k >= iters:
                active = active & cont.unsqueeze(0)
            if not bool(active.any()):
                if k < iters:
                    continue    # joint passes done: go to the containment check
                break
            _, J = self.margins_and_jacobian(state, a, offsets)
            stats["iters"] += 1
            stats["groups"].update(heads[c].group for c in active.any(0).nonzero().flatten().tolist())
            damp = damping if k < iters else damping * 1e-2
            a = self._correct(a, m, J * move, active, priority, target, damp)
            m = self.margins(state, a, offsets)
        return a, stats

    def _toward_fallback(self, state, a, m, offsets, n=8):
        """Line search toward the G3 fallback for rows the joint passes left short of G3.

        The fallback (no feed, slowest clock, growth-optimal light) is the
        action that keeps G3 most reachable and raises neither nitrate nor
        volume, so of n evenly spaced points on the segment toward it
        (stage-locked dims kept) the closest one every head accepts is taken,
        else the one with the best G3 margin.  One batched margin evaluation.

        Returns:
            A tuple of the actions [N, 4] and their margins [N, H].
        """
        names = [h.name for h in self.heads]
        if "g3" not in names:
            return a, m
        g3 = names.index("g3")
        short = m[:, g3] < 0.0
        if not bool(short.any()):
            return a, m
        rows = short.nonzero().flatten()
        s, x = state[rows].float(), a[rows]
        mask = PhycocyaninEnvCore.get_action_mask(s)
        target = torch.tensor(G3_FALLBACK, device=a.device) * mask + x * (1 - mask)
        lam = torch.linspace(1.0 / n, 1.0, n, device=a.device).view(n, 1, 1)
        cand = x.unsqueeze(0) + lam * (target - x).unsqueeze(0)                 # [n, R, 4]
        mc = self.margins(s.unsqueeze(0).expand(n, -1, -1), cand, offsets)      # [n, R, H]
        ok = (mc >= 0.0).all(-1)                                                # [n, R]
        first = torch.where(ok.any(0), ok.float().argmax(0), mc[..., g3].argmax(0))
        pick = torch.arange(len(rows), device=a.device)
        a, m = a.clone(), m.clone()
        a[rows] = cand[first, pick]
        m[rows] = mc[first, pick]
        return a, m

    # ── Safety Gradient ───────────────────────────────────────────────────────

    def safety_loss(self, state, action, sharpness=50.0, eps=PROJ_EPS):
        """Differentiable penalty on every head's shortfall (the actor's safety gradient).

        A smooth hinge, softplus(sharpness * (eps - m)) / sharpness, so safe
        margins contribute almost nothing and each violated head pulls the
        action along its own margin gradient.  Harvesting and Idle rows are
        excluded.  Use with frozen APN weights: the gradient reaches the actor
        only through ``action``.

        Args:
            state: Tensor [..., 12].
            action: Tensor [..., 4], a differentiable function of the actor.

        Returns:
            Scalar loss (mean over the active rows' heads).
        """
        m = self.margins(state, action)
        hinge = F.softplus(sharpness * (eps - m)) / sharpness          # [..., H]
        active = projection_active(state).to(hinge.dtype).unsqueeze(-1)
        return (hinge * active).sum() / (active.sum() * m.shape[-1]).clamp(min=1.0)


def load_apn(device):
    """The APN, frozen and in eval mode, with KineticNet loaded if it is trained.

    The exact heads need no checkpoint; without KINETIC_PATH the G3 head is
    left out (with a warning).
    """
    apn = MultiHeadAPN().to(device)
    if not apn.load_kinetic(map_location=device):
        print(f"[APN] {KINETIC_PATH} not found: G3 is not covered "
              "(train it with python -m safety.train).")
    apn.eval()
    for p in apn.parameters():
        p.requires_grad_(False)
    return apn
