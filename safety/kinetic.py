"""KineticNet: the network behind the APN's learned G3 head (as Fermentation-PPO's)."""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Local ────────────────────────────────────────────────────────────────────
from safety.constants import DRAWDOWN_STEPS

# Scales of the network's inputs and outputs (physical units).
CX_REF, CN_REF, DROP_REF = 6.0, 800.0, 50.0


class KineticNet(nn.Module):
    """Nitrate draw-down of the G3 fallback, control step by control step.

    Under the G3 fallback (no feed, growth-optimal light) nitrate only falls
    through uptake, and neither volume (the feed's dilution cancels it) nor
    product nor the clock (a control step is 10 h at any speed) enters the
    kinetics of biomass and nitrate.  So from the state after the current
    action, the fallback's nitrate at the end of each of the next
    DRAWDOWN_STEPS control steps is a function of biomass and nitrate alone:

        CN_k = max(CN - sum_{j<=k} drop_j, 0),   drop_j >= 0,

    monotone by construction.  Which step ends Growth is credit arithmetic,
    done exactly in the APN (safety/apn.py), not learned.
    """

    def __init__(self, hidden=128, steps=DRAWDOWN_STEPS):
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, steps),
        )

    def forward(self, cx, cn):
        """Fallback nitrate profile from biomass and nitrate.

        Args:
            cx: Biomass [...] (g/L).
            cn: Nitrate [...] (mg/L).

        Returns:
            Nitrate [..., DRAWDOWN_STEPS] (mg/L) at the end of each fallback step.
        """
        x = torch.stack([cx / CX_REF, cn / CN_REF], dim=-1)
        drop = F.softplus(self.net(x)) * DROP_REF
        return (cn.unsqueeze(-1) - drop.cumsum(-1)).clamp(min=0.0)
