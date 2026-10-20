"""The plant's kinetics in torch: what the APN's exact heads compute with.

envs/kinetics.py's rate equations, batched and differentiable in the action,
integrated by RK4 over one control step.  The exact heads (safety/apn.py) are
this one step, as Fermentation-PPO's exact heads are its plant's mass balance
over one hold; nothing here is learned.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from safety.constants import C_N_STOCK, CONTROL_INTERVAL

# envs/kinetics.py's parameters (no death, ud = 0).
_UM, _KN, _YNX, _KM, _KD = 0.0572, 393.1, 504.5, 0.00016, 0.281
_KS, _KI, _KSQ, _KIQ, _KNP = 178.9, 447.1, 23.51, 800.0, 16.89

# RK4 steps per control step.  The environment takes sixty 10-min ones; three
# stay within 0.03 mg/L of its nitrate and 0.3% of the Cq/Cx limit (the
# buffers are 2%), at a fraction of the cost of the projection.
SUBSTEPS = 3


def rates(x, I, Fn, Fout):
    """Rates of change [..., 4] of x = (Cx, CN, Cq, V), as envs/kinetics.calculate_rates_numba."""
    cx, cn, cq = x[..., 0], x[..., 1], x[..., 2]
    v = x[..., 3].clamp(min=0.1)
    f_in = (Fn * v / C_N_STOCK).clamp(min=0.0)
    phi_i = I / (I + _KS + I * I / _KI)
    phi_n = cn / (cn + _KN)
    phi_iq = I / (I + _KSQ + I * I / _KIQ)
    growth = _UM * phi_i * cx * phi_n
    return torch.stack([
        growth - f_in * cx / v,
        -_YNX * growth + f_in * (C_N_STOCK - cn) / v,
        _KM * phi_iq * cx - _KD * cq / (cn + _KNP) - f_in * cq / v,
        f_in - Fout,
    ], dim=-1)


def control_step(x, I, Fn, Fout, substeps=SUBSTEPS):
    """The physical state after one control step (CONTROL_INTERVAL hours) at constant inputs.

    Args:
        x: Physical state [..., 4] (Cx g/L, CN mg/L, Cq mg/L, V L).
        I, Fn, Fout: Light, nitrate feed and outflow [...] (physical units).
        substeps: RK4 steps.

    Returns:
        State [..., 4], clamped to the plant's floors like integrate_rk4.
    """
    h = CONTROL_INTERVAL / substeps
    floor = x.new_tensor([0.0, 0.0, 0.0, 0.1])
    for _ in range(substeps):
        k1 = rates(x, I, Fn, Fout)
        k2 = rates(x + 0.5 * h * k1, I, Fn, Fout)
        k3 = rates(x + 0.5 * h * k2, I, Fn, Fout)
        k4 = rates(x + h * k3, I, Fn, Fout)
        x = torch.maximum(x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), floor)
    return x
