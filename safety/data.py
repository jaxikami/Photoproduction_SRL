"""Training data for the APN's learned G3 head (apn.KineticNet).

The exact heads (G1, G2, G4) need no data: they are the plant's own one-step
kinetics (safety/dynamics.py).  The learned head predicts the G3 fallback's
nitrate draw-down (no feed, growth-optimal light) from biomass and nitrate, so
its data are (Cx, CN) pairs labelled by running the fallback on the same
kinetics for DRAWDOWN_STEPS control steps.
"""

import torch

from safety.constants import DRAWDOWN_STEPS, I_OPT, N_LIMIT_PATH
from safety.dynamics import control_step


def drawdown(cx, cn, steps=DRAWDOWN_STEPS):
    """The G3 fallback's nitrate at the end of each of the next ``steps`` control steps.

    Args:
        cx: Biomass [N] (g/L).
        cn: Nitrate [N] (mg/L).
        steps: Control steps.

    Returns:
        Nitrate [N, steps] (mg/L).
    """
    # Volume and product do not enter the biomass and nitrate kinetics
    # without feed; any value serves.
    x = torch.stack([cx, cn, torch.zeros_like(cx), torch.full_like(cx, 10.0)], dim=-1)
    I = torch.full_like(cx, I_OPT)
    zero = torch.zeros_like(cx)
    out = []
    for _ in range(steps):
        # Ten RK4 steps per control step: the labels follow the environment's
        # integrator to well under 0.01 mg/L over the whole draw-down.
        x = control_step(x, I, zero, zero, substeps=10)
        out.append(x[:, 1])
    return torch.stack(out, dim=1)


def make_dataset(n, seed=0, device="cpu"):
    """(Cx, CN) pairs and their fallback draw-down.

    Biomass is log-uniform over what the plant reaches; nitrate is uniform up
    to past the path limit, with a third of the samples where the draw-down
    ends near the G3 limit (mid nitrate) to sharpen the head there.

    Args:
        n: Pairs.
        seed: Seed.
        device: Torch device of the labelling.

    Returns:
        Dict with 'cx' [n], 'cn' [n] and 'profile' [n, DRAWDOWN_STEPS] tensors.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    cx = torch.exp(torch.empty(n).uniform_(torch.log(torch.tensor(0.05)).item(),
                                           torch.log(torch.tensor(9.0)).item(), generator=g))
    cn = torch.rand(n, generator=g) * N_LIMIT_PATH * 1.25
    mid = torch.rand(n, generator=g) < 1.0 / 3.0
    cn[mid] = 100.0 + torch.rand(int(mid.sum()), generator=g) * 500.0
    cx, cn = cx.to(device, torch.float64), cn.to(device, torch.float64)
    profile = drawdown(cx, cn)
    return {"cx": cx.float(), "cn": cn.float(), "profile": profile.float()}
