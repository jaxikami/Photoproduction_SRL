"""
Validation suite for the Photoproduction Safety Boundary Classifier.

Tests the gradient-ascent projection (APN) against instantaneous constraints:
  - Test 1 (G1): Nitrate path constraint
  - Test 2 (G2): Product/biomass ratio
  - Test 3 (G4): Reactor overflow
  - Test 4 (ID): Identity mapping (safe states unmodified)

Note: G3 (terminal nitrate) and G5 (idle stage) are handled by the GRU temporal 
context and Lagrangian multipliers, not by the APN's instantaneous projection.
"""

import numpy as np
import torch

from envs.kinetics import integrate_rk4
from envs.plant import DEFAULT_SQUASHED, PhycocyaninEnvCore
from safety.apn import load_apn
from safety.constants import (C_N_STOCK, CONTROL_INTERVAL, FN_MAX_GROWTH, FN_MAX_PROD, I_MAX,
                              I_MIN, N_LIMIT_PATH, RATIO_LIMIT, SAFE_BUFFER, V_MAX, V_MIN)


def _project_to_safe(apn, state_norm_t, action_t):
    """The safe agent's L0 → L2 → L0 pipeline on one action.

    Stage-masks the action, projects it with the APN and re-applies the mask,
    exactly as agents/safe.py does at every control step.

    Args:
        apn (ActionProjectionNetwork): The loaded APN model.
        state_norm_t (torch.Tensor): The current normalized state observation.
        action_t (torch.Tensor): The proposed action in [-1, 1].

    Returns:
        tuple: A tuple containing:
            - projected_action (torch.Tensor): The projected, safe action tensor.
            - optimization_steps_taken (int): The number of gradient steps taken.
    """
    stage_mask = PhycocyaninEnvCore.get_action_mask(state_norm_t)
    default = torch.tensor(DEFAULT_SQUASHED, device=state_norm_t.device)
    a = action_t * stage_mask + default * (1 - stage_mask)
    a, stats = apn.project(state_norm_t, a)
    return a * stage_mask + default * (1 - stage_mask), stats["iters"]


def _make_state(cx, cN, cq, V, stage_idx, credit_norm, t_norm, supply_norm, device):
    """Packs physical values into a normalized state tensor shape [1, 12].

    Args:
        cx (float): Biomass concentration (g/L).
        cN (float): Nitrate concentration (mg/L).
        cq (float): Phycocyanin concentration (mg/L).
        V (float): Reactor volume (L).
        stage_idx (int): Current operational stage index [0-3].
        credit_norm (float): Normalized remaining stage credit.
        t_norm (float): Normalized episode time.
        supply_norm (float): Normalized remaining nitrate supply.
        device (torch.device): Compute device for the output tensor.

    Returns:
        torch.Tensor: Normalized observation tensor.
    """
    s = torch.zeros(1, 12, dtype=torch.float32, device=device)
    s[0, 0] = cx / 6.0
    s[0, 1] = cN / 800.0
    s[0, 2] = cq / 0.2
    s[0, 3] = V / V_MAX
    s[0, 4 + stage_idx] = 1.0
    s[0, 8] = credit_norm
    s[0, 9] = t_norm
    s[0, 10] = supply_norm
    s[0, 11] = 1.0 - t_norm  # operation time left
    return s


def run_validation(model=None, num_test_samples: int = 2000):
    """Executes the constraint validation suite using physics-based simulations.

    Validates that the APN effectively blocks unsafe actions from breaching
    the defined hard bounds for the G1, G2, and G4 constraints, and verifies
    that safe actions remain undisturbed.

    Args:
        model (ActionProjectionNetwork, optional): Pre-loaded APN. If None, it will
            be loaded from the policy directory. Defaults to None.
        num_test_samples (int, optional): Number of randomized samples per test.
            Defaults to 2000.

    Returns:
        tuple:
            - all_passed (bool): True if all tests exceed the 95% threshold.
            - pass_rates (dict): Dictionary mapping constraint names to success rates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        apn = load_apn(device)
        if apn is None:
            print("[Validation] ERROR: No checkpoint found. Run python -m safety.train first.")
            return False, {}
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
        if _ % 2 == 0:
            # Edge case: high nitrate, normal volume
            cN = float(torch.empty(1).uniform_(N_LIMIT_PATH * 0.85, N_LIMIT_PATH * 1.00))
            cx = float(torch.empty(1).uniform_(0.5, 5.0))
            cq = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.8))
            V  = float(torch.empty(1).uniform_(V_MAX * 0.60, V_MAX * 0.90))
            t  = float(torch.empty(1).uniform_(0.0, 0.6))
        else:
            # Broad case: full valid spectrum
            cN = float(torch.empty(1).uniform_(0.0, N_LIMIT_PATH * 1.00))
            cx = float(torch.empty(1).uniform_(0.1, 6.0))
            cq = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.9))
            V  = float(torch.empty(1).uniform_(V_MIN, V_MAX * 0.95))
            t  = float(torch.empty(1).uniform_(0.0, 0.9))

        s_t = _make_state(cx, cN, cq, V, stage_idx=0, credit_norm=0.5,
                          t_norm=t, supply_norm=0.5, device=device)
        a_t = torch.ones(1, 4, device=device)  # max intent

        state_phys = np.array([cx, cN, cq, V], dtype=np.float64)
        rollout_steps = 5
        for step_idx in range(rollout_steps):
            a_proj, iters = _project_to_safe(apn, s_t, a_t)
            g1_iters.append(iters)
            
            a_scaled = (a_proj[0] + 1.0) / 2.0
            I_phys = I_MIN + a_scaled[1].item() * (I_MAX - I_MIN)
            Fn_phys = a_scaled[2].item() * FN_MAX_GROWTH
            
            state_phys = integrate_rk4(state_phys, I_phys, Fn_phys, 0.0, 10.0/60.0, 60)
            
            if step_idx < rollout_steps - 1:
                s_t = _make_state(state_phys[0], state_phys[1], state_phys[2], state_phys[3], stage_idx=0, credit_norm=0.5,
                                  t_norm=t, supply_norm=0.5, device=device)

        if state_phys[1] <= N_LIMIT_PATH * SAFE_BUFFER:
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
        if _ % 2 == 0:
            # Edge case: ratio near limit
            cx = float(torch.empty(1).uniform_(0.5, 5.0))
            cq = cx * RATIO_LIMIT * float(torch.empty(1).uniform_(0.90, 1.00))
            cN = float(torch.empty(1).uniform_(350.0, 600.0))
            V  = float(torch.empty(1).uniform_(V_MAX * 0.60, V_MAX * 0.90))
            t  = float(torch.empty(1).uniform_(0.0, 0.6))
        else:
            # Broad case: full valid spectrum
            cx = float(torch.empty(1).uniform_(0.1, 6.0))
            cq = cx * RATIO_LIMIT * float(torch.empty(1).uniform_(0.0, 1.00))
            cN = float(torch.empty(1).uniform_(0.0, 800.0))
            V  = float(torch.empty(1).uniform_(V_MIN, V_MAX * 0.95))
            t  = float(torch.empty(1).uniform_(0.0, 0.9))

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
        if _ % 2 == 0:
            # Edge case: volume near limit
            V  = float(torch.empty(1).uniform_(V_MAX * 0.85, V_MAX * SAFE_BUFFER))
            cx = float(torch.empty(1).uniform_(0.5, 5.0))
            cN = float(torch.empty(1).uniform_(0.0, 400.0))
            cq = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.8))
            t  = float(torch.empty(1).uniform_(0.0, 0.5))
        else:
            # Broad case: full valid spectrum
            V  = float(torch.empty(1).uniform_(V_MIN, V_MAX * SAFE_BUFFER))
            cx = float(torch.empty(1).uniform_(0.1, 6.0))
            cN = float(torch.empty(1).uniform_(0.0, 800.0))
            cq = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.8))
            t  = float(torch.empty(1).uniform_(0.0, 0.9))

        stage = 0 if torch.rand(1).item() > 0.5 else 1
        s_t = _make_state(cx, cN, cq, V, stage_idx=stage, credit_norm=0.5,
                          t_norm=t, supply_norm=0.5, device=device)
        a_t = torch.ones(1, 4, device=device)  # max feed → max volume increase

        a_proj, iters = _project_to_safe(apn, s_t, a_t)
        g4_iters.append(iters)

        # Physics
        a_scaled = (a_proj[0] + 1.0) / 2.0
        fn_max = FN_MAX_GROWTH if stage == 0 else FN_MAX_PROD
        Fn_phys = a_scaled[2].item() * fn_max
        Fout_phys = 0.0  # masked in growth or inoculation
        
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
        if _ % 2 == 0:
            # Typical operation states
            cx     = float(torch.empty(1).uniform_(0.5, 5.0))
            cN     = float(torch.empty(1).uniform_(0.0, N_LIMIT_PATH * 0.50))
            cq     = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.60))
            V      = float(torch.empty(1).uniform_(V_MAX * 0.40, V_MAX * 0.80))
            t_norm = float(torch.empty(1).uniform_(0.0, 0.50))
        else:
            # Broad safe spectrum
            cx     = float(torch.empty(1).uniform_(0.1, 6.0))
            cN     = float(torch.empty(1).uniform_(0.0, N_LIMIT_PATH * 0.80))
            cq     = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.80))
            V      = float(torch.empty(1).uniform_(V_MIN, V_MAX * 0.90))
            t_norm = float(torch.empty(1).uniform_(0.0, 0.90))

        s_t = _make_state(cx, cN, cq, V, stage_idx=0, credit_norm=0.7,
                          t_norm=t_norm, supply_norm=0.7, device=device)
        a_t = -0.5 + torch.rand(1, 4, device=device)  # moderate safe intent
        
        # Pre-mask a_t so we don't unfairly penalize the default clamp
        stage_mask = PhycocyaninEnvCore.get_action_mask(s_t)
        default_squashed = torch.tensor(DEFAULT_SQUASHED, device=device)
        a_t_masked = a_t * stage_mask + default_squashed * (1 - stage_mask)

        a_proj, _ = _project_to_safe(apn, s_t, a_t_masked)
        diff = (a_proj - a_t_masked).abs().max().item()
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