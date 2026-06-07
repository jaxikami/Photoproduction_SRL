"""
Validation suite for the Photoproduction Safety Boundary Classifier.

Uses volume-less kinetics: concentrations evolve without dilution terms.
F_out (well-mixed drain) does not change concentrations.

Tests the gradient-ascent projection (APN) against instantaneous constraints:
  - Test 1 (G1): Nitrate path constraint
  - Test 2 (G2): Product/biomass ratio
  - Test 3 (G4): Total mass concentration (M_total = Cx*1000 + CN + Cq ≤ M_CONC_LIMIT)
  - Test 4 (ID): Identity mapping (safe states unmodified)

Note: G3 (terminal nitrate) and G5 (idle stage) are handled by the GRU temporal 
context and Lagrangian multipliers, not by the APN's instantaneous projection.
"""

import torch
import numpy as np
import os
from pretrain import ActionProjectionNetwork, load_compatible_checkpoint
from env_core import PhycocyaninEnvCore

# ── Physical constants (must match env.py and data_gen.py) ────────────────────
I_MIN, I_MAX     = 120.0, 400.0
FN_MAX_GROWTH    = 40.0
FN_MAX_PROD      = 10.0
FOUT_MAX         = 0.05         # h^-1 fractional drain rate (not used in kinetics)
N_LIMIT_PATH     = 800.0
RATIO_LIMIT      = 0.011

M_CONC_LIMIT     = 5000.0       # mg/L total mass concentration limit
CONTROL_INTERVAL = 10.0
TOTAL_TIME       = 1000.0
SAFE_BUFFER      = 0.98
THRESHOLD        = 0.711
MAX_PROJ_STEPS   = 7
LR_PROJ          = 0.5


def _project_to_safe(apn, state_norm_t, action_t, max_steps=MAX_PROJ_STEPS, lr=LR_PROJ, threshold=THRESHOLD):
    """Executes gradient-ascent projection to find a safe action proxy.
    
    Mirrors the safe_agent.SPRL_Agent._project_to_safe logic for validation.
    Includes full SERL masking logic to prevent the APN from cheating
    using structurally inactive flow dimensions.

    Args:
        apn (ActionProjectionNetwork): The loaded APN model.
        state_norm_t (torch.Tensor): The current normalized state observation.
        action_t (torch.Tensor): The unbounded proposed action.
        max_steps (int, optional): Max optimization iterations. Defaults to MAX_PROJ_STEPS.
        lr (float, optional): Gradient ascent learning rate. Defaults to LR_PROJ.
        threshold (float, optional): Target safety probability. Defaults to THRESHOLD.

    Returns:
        tuple: (projected_action (torch.Tensor), optimization_steps_taken (int))
    """
    state_fixed = state_norm_t.detach()
    from env_safe import PhycocyaninEnvSafe
    stage_mask = PhycocyaninEnvSafe.get_action_mask(state_fixed)
    default_squashed = torch.tensor([-0.333, -1.0, -1.0, -1.0], device=state_fixed.device)

    # SERL checkpoint 2: clamp initial action
    a = action_t.clone().detach() * stage_mask + default_squashed * (1 - stage_mask)

    # Bypass projection in Stage 2 (Harvesting) and Stage 3 (Idle) to avoid phantom gradients
    is_stage_2_or_3 = (state_fixed[..., 6] > 0.5) | (state_fixed[..., 7] > 0.5)
    if is_stage_2_or_3.any():
        return a, 0

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
                a_final = a_var.detach()
                return a_final * stage_mask + default_squashed * (1 - stage_mask), step

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

                # Constant step size with mild decay — avoids premature stall
                step_size = lr / (1.0 + step * 0.03)
                a = a + step_size * grad.sign()
                a = a.clamp(-1.0, 1.0)

    # SERL checkpoint 3
    best_a = best_a * stage_mask + default_squashed * (1 - stage_mask)
    return best_a, max_steps


def _make_state(cx, cN, cq, stage_idx, credit_norm, t_norm, supply_norm, device):
    """Packs physical values into a normalized state tensor shape [1, 12].

    M_total (= cx*1000 + cN + cq) is derived from the other concentrations
    and normalised by M_CONC_LIMIT.

    Args:
        cx (float): Biomass concentration (g/L).
        cN (float): Nitrate concentration (mg/L).
        cq (float): Phycocyanin concentration (mg/L).
        stage_idx (int): Current operational stage index [0-3].
        credit_norm (float): Normalized remaining stage credit.
        t_norm (float): Normalized episode time.
        supply_norm (float): Normalized remaining nitrate supply.
        device (torch.device): Compute device for the output tensor.

    Returns:
        torch.Tensor: Normalized observation tensor.
    """
    M_total = cx * 1000.0 + cN + cq
    s = torch.zeros(1, 12, dtype=torch.float32, device=device)
    s[0, 0] = cx / 6.0
    s[0, 1] = cN / 800.0
    s[0, 2] = cq / 0.2
    s[0, 3] = M_total / M_CONC_LIMIT
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
        for ckpt in [os.path.join("policy", "action_projection_network.pth"),
                     "action_projection_network.pth"]:
            if os.path.exists(ckpt):
                apn = ActionProjectionNetwork.from_checkpoint(
                    ckpt, device, state_dim=12, action_dim=4)
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
        t  = float(torch.empty(1).uniform_(0.0, 0.6))

        s_t = _make_state(cx, cN, cq, stage_idx=0, credit_norm=0.5,
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
        # Volume-less kinetics: Fn enters directly, no F_in_vol
        dCN = -YNX * growth + Fn_phys
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
        t  = float(torch.empty(1).uniform_(0.0, 0.6))

        s_t = _make_state(cx, cN, cq, stage_idx=1, credit_norm=0.5,
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
        # Volume-less kinetics: no dilution terms
        dCx = growth
        dCq = km * phi_Iq * cx - (kd * cq) / (cN + KNp)

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



    print(f"\n--- [TEST 3] G4: Mass concentration (M_total ≤ {M_CONC_LIMIT*SAFE_BUFFER:.0f} mg/L) ---")
    g4_passes = 0
    g4_iters  = []

    for _ in range(num_test_samples):
        cx = float(torch.empty(1).uniform_(3.0, 4.2))
        cN = float(torch.empty(1).uniform_(200.0, 400.0))
        cq = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.8))
        
        M_init = cx * 1000.0 + cN + cq
        if M_init >= M_CONC_LIMIT * SAFE_BUFFER:
            continue
            
        t  = float(torch.empty(1).uniform_(0.0, 0.5))

        s_t = _make_state(cx, cN, cq, stage_idx=0, credit_norm=0.5,
                          t_norm=t, supply_norm=0.5, device=device)
        a_t = torch.ones(1, 4, device=device)  # max feed → max mass increase

        a_proj, iters = _project_to_safe(apn, s_t, a_t)
        g4_iters.append(iters)

        # Physics — volume-less kinetics
        a_scaled = (a_proj[0] + 1.0) / 2.0
        I_phys = I_MIN + a_scaled[1].item() * (I_MAX - I_MIN)
        Fn_phys = a_scaled[2].item() * FN_MAX_GROWTH

        um = 0.0572; KN = 393.1; YNX = 504.5; km = 0.00016; kd = 0.281
        ks = 178.9; ki = 447.1; ksq = 23.51; kiq = 800.0; KNp = 16.89

        phi_I = I_phys / (I_phys + ks + (I_phys**2 / ki))
        phi_N = cN / (cN + KN)
        phi_Iq = I_phys / (I_phys + ksq + (I_phys**2 / kiq))
        growth = um * phi_I * cx * phi_N

        dCx = growth
        dCN = -YNX * growth + Fn_phys
        dCq = km * phi_Iq * cx - (kd * cq) / (cN + KNp)

        cx_next = cx + dCx * CONTROL_INTERVAL
        cN_next = cN + dCN * CONTROL_INTERVAL
        cq_next = cq + dCq * CONTROL_INTERVAL
        M_total_next = cx_next * 1000.0 + cN_next + cq_next

        if M_total_next <= M_CONC_LIMIT * SAFE_BUFFER:
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
        cx     = float(torch.empty(1).uniform_(0.5, 4.0))   # cap at 4.0
        cN     = float(torch.empty(1).uniform_(0.0, N_LIMIT_PATH * 0.50))
        cq     = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.60))
        
        # G4 pre-screen: ensure state + minimum growth stays under limit
        M_init = cx * 1000.0 + cN + cq
        phi_I_min = I_MIN / (I_MIN + 178.9 + I_MIN**2 / 447.1)
        phi_N_val = cN / (cN + 393.1) if cN > 0 else 0.0
        dM_min = (1000.0 - 504.5) * 0.0572 * phi_I_min * cx * phi_N_val * CONTROL_INTERVAL
        if M_init + dM_min >= M_CONC_LIMIT * SAFE_BUFFER:
            continue
            
        t_norm = float(torch.empty(1).uniform_(0.0, 0.50))

        s_t = _make_state(cx, cN, cq, stage_idx=0, credit_norm=0.7,
                          t_norm=t_norm, supply_norm=0.7, device=device)
        a_t = -0.5 + torch.rand(1, 4, device=device)  # moderate safe intent
        
        # Pre-mask a_t so we don't unfairly penalize the default clamp
        from env_safe import PhycocyaninEnvSafe
        stage_mask = PhycocyaninEnvSafe.get_action_mask(s_t)
        default_squashed = torch.tensor([-0.333, -1.0, -1.0, -1.0], device=device)
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