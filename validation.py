"""
Validation suite for the Photoproduction Safety Boundary Classifier.

Tests the gradient-ascent projection against four instantaneous constraints:
  Test 1 — G1: Nitrate path constraint
  Test 2 — G2: Product/biomass ratio
  Test 4 — Identity mapping (safe states unmodified)

Note: G3 (terminal nitrate) is handled by the GRU temporal context and
Lagrangian multiplier, not by the APN.
"""

import torch
import numpy as np
import os
from pretrain import ActionProjectionNetwork

# ── Physical constants (must match env.py and data_gen.py) ────────────────────
I_MIN, I_MAX     = 120.0, 400.0
FN_MAX_GROWTH    = 40.0
FN_MAX_PROD      = 10.0
FOUT_MAX         = 2.0
N_LIMIT_PATH     = 800.0
RATIO_LIMIT      = 0.011

V_MAX            = 50.0
V_MIN            = 5.0
C_N_STOCK        = 50000.0
CONTROL_INTERVAL = 10.0
TOTAL_TIME       = 500.0
SAFE_BUFFER      = 0.98
THRESHOLD        = 0.95
MAX_PROJ_STEPS   = 5
LR_PROJ          = 1.0


def _project_to_safe(apn, state_norm_t, action_t, max_steps=MAX_PROJ_STEPS,
                      lr=LR_PROJ, threshold=THRESHOLD):
    """
    Gradient-ascent projection — mirrors safe_agent.SPRL_Agent._project_to_safe.

    Uses constant step size (no severity decay) to ensure the projection
    reliably reaches the safe manifold within the budget.
    """
    a = action_t.clone().detach()
    state_fixed = state_norm_t.detach()

    with torch.no_grad():
        p = apn.classify(state_fixed, a)
        if p.item() >= threshold:
            return a, 0

    best_a      = a.clone()
    best_margin = apn(state_fixed, a).item()

    with torch.enable_grad():
        for step in range(max_steps):
            a_var  = a.clone().requires_grad_(True)
            margin = apn(state_fixed, a_var)

            p = torch.sigmoid(margin)
            if p.item() >= threshold:
                return a_var.detach(), step

            # Track best iterate seen
            m_val = margin.item()
            if m_val > best_margin:
                best_margin = m_val
                best_a      = a_var.detach().clone()

            grad = torch.autograd.grad(margin.sum(), a_var)[0]

            with torch.no_grad():
                grad = grad.clone()
                at_lower = (a_var.data <= -0.9999) & (grad < 0)
                at_upper = (a_var.data >=  0.9999) & (grad > 0)
                grad[at_lower | at_upper] = 0.0

                # Constant step size with mild decay — avoids premature stall
                step_size = lr / (1.0 + step * 0.1)
                a = a + step_size * grad.sign()
                a = a.clamp(-1.0, 1.0)

    return best_a, max_steps


def _make_state(cx, cN, cq, V, stage_idx, credit_norm, t_norm, supply_norm, device):
    """Pack physical values into normalised state tensor [1, 11]."""
    s = torch.zeros(1, 11, dtype=torch.float32, device=device)
    s[0, 0] = cx / 6.0
    s[0, 1] = cN / 800.0
    s[0, 2] = cq / 0.2
    s[0, 3] = V / V_MAX
    s[0, 4 + stage_idx] = 1.0
    s[0, 8] = credit_norm
    s[0, 9] = t_norm
    s[0, 10] = supply_norm
    return s


def run_validation(model=None, num_test_samples: int = 2000):
    """
    Runs the five constraint validation tests.

    Returns:
        bool — True if all tests pass (≥ 95% projection success rate).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        apn = ActionProjectionNetwork(state_dim=11, action_dim=4).to(device)
        for ckpt in [os.path.join("policy", "action_projection_network.pth"),
                     "action_projection_network.pth"]:
            if os.path.exists(ckpt):
                apn.load_state_dict(torch.load(ckpt, map_location=device,
                                                weights_only=True))
                print(f"[Validation] Loaded APN from '{ckpt}'.")
                break
        else:
            print("[Validation] ERROR: No checkpoint found. Run pretrain.py first.")
            return False
    else:
        apn = model

    apn.eval()
    for p in apn.parameters():
        p.requires_grad_(False)

    PASS_THRESHOLD = 0.95
    all_passed = True

    # =========================================================================
    # TEST 1 — G1: Nitrate path constraint
    # =========================================================================
    print(f"\n--- [TEST 1] G1: Nitrate path (target < {N_LIMIT_PATH*SAFE_BUFFER:.0f} mg/L) ---")
    g1_passes = 0
    g1_iters  = []

    for _ in range(num_test_samples):
        cN = float(torch.empty(1).uniform_(N_LIMIT_PATH * 0.85, N_LIMIT_PATH * 1.00))
        cx = float(torch.empty(1).uniform_(0.5, 5.0))
        cq = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.8))
        V  = float(torch.empty(1).uniform_(30.0, 45.0))
        t  = float(torch.empty(1).uniform_(0.0, 0.6))

        s_t = _make_state(cx, cN, cq, V, stage_idx=0, credit_norm=0.5,
                          t_norm=t, supply_norm=0.5, device=device)
        a_t = torch.ones(1, 4, device=device)  # max intent

        a_proj, iters = _project_to_safe(apn, s_t, a_t)
        g1_iters.append(iters)

        # Physics
        a_scaled = (a_proj[0] + 1.0) / 2.0
        I_phys = I_MIN + a_scaled[1].item() * (I_MAX - I_MIN)
        Fn_phys = a_scaled[2].item() * FN_MAX_GROWTH
        
        um = 0.0572; KN = 393.1; YNX = 504.5; ks = 178.9; ki = 447.1
        phi_I = I_phys / (I_phys + ks + (I_phys**2 / ki))
        phi_N = cN / (cN + KN)
        growth = um * phi_I * cx * phi_N
        F_in_vol = Fn_phys * V / C_N_STOCK
        dCN = -YNX * growth + F_in_vol * (C_N_STOCK - cN) / V
        cN_next = cN + dCN * CONTROL_INTERVAL

        if cN_next <= N_LIMIT_PATH * SAFE_BUFFER:
            g1_passes += 1

    g1_rate = g1_passes / num_test_samples
    print(f"  Pass rate: {g1_passes}/{num_test_samples} ({g1_rate:.1%})  "
          f"Avg iters: {np.mean(g1_iters):.1f}")
    if g1_rate < PASS_THRESHOLD:
        print("  ❌ FAILED")
        all_passed = False
    else:
        print("  ✓ PASSED")

    # =========================================================================
    # TEST 2 — G2: Product/biomass ratio
    # =========================================================================
    print(f"\n--- [TEST 2] G2: Product/biomass ratio (cq/cx ≤ {RATIO_LIMIT*SAFE_BUFFER:.4f}) ---")
    g2_passes = 0
    g2_iters  = []

    for _ in range(num_test_samples):
        cx = float(torch.empty(1).uniform_(0.5, 5.0))
        cq = cx * RATIO_LIMIT * float(torch.empty(1).uniform_(0.90, 1.00))
        cN = float(torch.empty(1).uniform_(350.0, 600.0))
        V  = float(torch.empty(1).uniform_(30.0, 45.0))
        t  = float(torch.empty(1).uniform_(0.0, 0.6))

        s_t = _make_state(cx, cN, cq, V, stage_idx=1, credit_norm=0.5,
                          t_norm=t, supply_norm=0.5, device=device)
        a_t = torch.ones(1, 4, device=device)

        a_proj, iters = _project_to_safe(apn, s_t, a_t)
        g2_iters.append(iters)

        # Physics
        a_scaled = (a_proj[0] + 1.0) / 2.0
        I_phys = I_MIN + a_scaled[1].item() * (I_MAX - I_MIN)
        Fn_phys = a_scaled[2].item() * FN_MAX_PROD

        um = 0.0572; KN = 393.1; km = 0.00016; kd = 0.281
        ks = 178.9; ki = 447.1; ksq = 23.51; kiq = 800.0; KNp = 16.89

        phi_I = I_phys / (I_phys + ks + (I_phys**2 / ki))
        phi_N = cN / (cN + KN)
        phi_Iq = I_phys / (I_phys + ksq + (I_phys**2 / kiq))

        growth = um * phi_I * cx * phi_N
        F_in_vol = Fn_phys * V / C_N_STOCK

        dCx = growth - (F_in_vol * cx / V)
        dCq = km * phi_Iq * cx - (kd * cq) / (cN + KNp) - (F_in_vol * cq / V)

        cx_next = cx + dCx * CONTROL_INTERVAL
        cq_next = cq + dCq * CONTROL_INTERVAL
        ratio_next = cq_next / (cx_next + 1e-8)

        if ratio_next <= RATIO_LIMIT * SAFE_BUFFER:
            g2_passes += 1

    g2_rate = g2_passes / num_test_samples
    print(f"  Pass rate: {g2_passes}/{num_test_samples} ({g2_rate:.1%})  "
          f"Avg iters: {np.mean(g2_iters):.1f}")
    if g2_rate < PASS_THRESHOLD:
        print("  ❌ FAILED")
        all_passed = False
    else:
        print("  ✓ PASSED")



    print(f"\n--- [TEST 3] G4: Reactor overflow (V ≤ {V_MAX*SAFE_BUFFER:.0f} L) ---")
    g4_passes = 0
    g4_iters  = []

    for _ in range(num_test_samples):
        V  = float(torch.empty(1).uniform_(V_MAX * 0.85, V_MAX * SAFE_BUFFER))
        cx = float(torch.empty(1).uniform_(0.5, 5.0))
        cN = float(torch.empty(1).uniform_(0.0, 400.0))
        cq = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.8))
        t  = float(torch.empty(1).uniform_(0.0, 0.5))

        s_t = _make_state(cx, cN, cq, V, stage_idx=0, credit_norm=0.5,
                          t_norm=t, supply_norm=0.5, device=device)
        a_t = torch.ones(1, 4, device=device)  # max feed → max volume increase

        a_proj, iters = _project_to_safe(apn, s_t, a_t)
        g4_iters.append(iters)

        # Physics
        a_scaled = (a_proj[0] + 1.0) / 2.0
        Fn_phys = a_scaled[2].item() * FN_MAX_GROWTH
        Fout_phys = 0.0  # masked in growth
        
        F_in_vol = Fn_phys * V / C_N_STOCK
        V_next = V + (F_in_vol - Fout_phys) * CONTROL_INTERVAL

        if V_next <= V_MAX * SAFE_BUFFER:
            g4_passes += 1

    g4_rate = g4_passes / num_test_samples
    print(f"  Pass rate: {g4_passes}/{num_test_samples} ({g4_rate:.1%})  "
          f"Avg iters: {np.mean(g4_iters):.1f}")
    if g4_rate < PASS_THRESHOLD:
        print("  ❌ FAILED")
        all_passed = False
    else:
        print("  ✓ PASSED")

    print(f"\n--- [TEST 4] Identity mapping: safe states unmodified (diff ≤ 2%) ---")
    id_passes = 0
    for _ in range(num_test_samples):
        cx     = float(torch.empty(1).uniform_(0.5, 5.0))
        cN     = float(torch.empty(1).uniform_(0.0, N_LIMIT_PATH * 0.50))
        cq     = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.60))
        V      = float(torch.empty(1).uniform_(20.0, 40.0))
        t_norm = float(torch.empty(1).uniform_(0.0, 0.50))

        s_t = _make_state(cx, cN, cq, V, stage_idx=0, credit_norm=0.7,
                          t_norm=t_norm, supply_norm=0.7, device=device)
        a_t = -0.5 + torch.rand(1, 4, device=device)  # moderate safe intent

        a_proj, _ = _project_to_safe(apn, s_t, a_t)
        diff = (a_proj - a_t).abs().max().item()
        if diff <= 0.02:
            id_passes += 1

    id_rate = id_passes / num_test_samples
    print(f"  Pass rate: {id_passes}/{num_test_samples} ({id_rate:.1%})")
    if id_rate < PASS_THRESHOLD:
        print("  ❌ FAILED")
        all_passed = False
    else:
        print("  ✓ PASSED")

    print(f"\n{'[ALL TESTS PASSED]' if all_passed else '[SOME TESTS FAILED]'}")
    return all_passed, {"G1": g1_rate, "G2": g2_rate, "G4": g4_rate}


if __name__ == "__main__":
    run_validation()