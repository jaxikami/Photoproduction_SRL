"""
Dataset generation for the Safety Boundary Classifier (Photoproduction).

Generates (state, action, margin) tuples covering 4 instantaneous constraints:
  G1 — Nitrate path:       cN + Fn*10h  ≤  N_LIMIT_PATH * SAFE_BUFFER
  G2 — Product ratio:      cq / cx       ≤  RATIO_LIMIT * SAFE_BUFFER
  G4 — Reactor overflow:   V + dV*10h    ≤  V_MAX * SAFE_BUFFER

Note: G3 (terminal nitrate, cN_final ≤ 150 mg/L) is a temporal constraint
handled by the GRU's temporal context and the Lagrangian multiplier, not
by the single-step APN safety filter.
"""

import torch

# ── Global safety parameters (mirror env.py) ─────────────────────────────────
I_MIN, I_MAX    = 120.0, 400.0
FN_MAX_GROWTH   = 40.0
FN_MAX_PROD     = 10.0
FOUT_MAX        = 2.0
N_LIMIT_PATH    = 800.0
RATIO_LIMIT     = 0.011

V_MAX           = 50.0
V_MIN           = 5.0
V_RESET         = 0.10 * V_MAX   # 5.0 L — post-harvest partial reset (10% V_MAX)
C_N_STOCK       = 50000.0
CONTROL_INTERVAL = 10.0
TOTAL_TIME      = 1000.0
SAFE_BUFFER     = 0.98


def _generate_raw_batch(num_samples: int, bias: float = 0.7, pass_rates: dict = None):
    """Generates a raw batch of state-action tuples and physical margins.

    Simulates the system dynamics forward by one step to compute instantaneous
    margins for the G1, G2, and G4 constraints. The dataset is biased toward
    generating states near the constraint boundaries to train the APN effectively.

    State layout (12D, normalized):
      [cx/6, cN/800, cq/0.2, V/V_MAX, stage_0..3, credit/base, t/500, supply/initial, op_time_left]

    Action layout (4D, [-1,1]):
      [time_mult, I_norm, Fn_norm, Fout_norm]

    Args:
        num_samples (int): Number of raw samples to generate.
        bias (float, optional): Fraction of samples dedicated to boundary regions. Defaults to 0.7.
        pass_rates (dict, optional): Dictionary of validation pass rates used to dynamically 
            adjust boundary sub-sampling priorities. Defaults to None.

    Returns:
        tuple: (states, actions, is_safe, smooth_min_margin, margin_g1, margin_g2, margin_g4, near_boundary_mask)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_postcycle = int(num_samples * 0.12)
    n_interior = int(num_samples * (1.0 - bias))
    n_boundary = num_samples - n_interior - n_postcycle

    if pass_rates is not None:
        err_g1 = max(0.0, 1.0 - pass_rates.get("G1", 1.0))
        err_g2 = max(0.0, 1.0 - pass_rates.get("G2", 1.0))
        err_g4 = max(0.0, 1.0 - pass_rates.get("G4", 1.0))
        total_err = err_g1 + err_g2 + err_g4
        if total_err > 1e-8:
            w_g1 = 0.2 * 0.35 + 0.8 * (err_g1 / total_err)
            w_g2 = 0.2 * 0.30 + 0.8 * (err_g2 / total_err)
            w_g4 = 0.2 * 0.35 + 0.8 * (err_g4 / total_err)
        else:
            w_g1, w_g2, w_g4 = 0.35, 0.30, 0.35
    else:
        w_g1, w_g2, w_g4 = 0.35, 0.30, 0.35

    n_g2 = int(n_boundary * w_g2)
    n_g4 = int(n_boundary * w_g4)
    n_g1 = n_boundary - n_g2 - n_g4

    # ══════════════════════════════════════════════════════════════════════════
    # INTERIOR samples — safely in the middle of operating space
    # ══════════════════════════════════════════════════════════════════════════
    cx_int = 0.5 + torch.rand(n_interior, device=device) * 5.0
    cN_int = torch.rand(n_interior, device=device) * N_LIMIT_PATH * 0.6
    cq_int = torch.rand(n_interior, device=device) * cx_int * RATIO_LIMIT * 0.5
    V_int  = 15.0 + torch.rand(n_interior, device=device) * 25.0  # 15-40 L
    t_int  = torch.rand(n_interior, device=device) * 0.7
    # Actions: moderate, mostly safe
    a_int  = torch.rand(n_interior, 4, device=device) * 1.4 - 0.7  # [-0.7, 0.7]

    # ══════════════════════════════════════════════════════════════════════════
    # G1 boundary — high cN with variable Fn feed
    # ══════════════════════════════════════════════════════════════════════════
    cx_g1 = 0.5 + torch.rand(n_g1, device=device) * 5.0
    # cN from 75% to 110% of limit — tighter focus on the dangerous upper tail
    cN_g1 = N_LIMIT_PATH * (0.75 + torch.rand(n_g1, device=device) * 0.35)
    cq_g1 = torch.rand(n_g1, device=device) * cx_g1 * RATIO_LIMIT * 0.5
    V_g1  = 20.0 + torch.rand(n_g1, device=device) * 25.0
    t_g1  = torch.rand(n_g1, device=device) * 0.8
    # Actions: sweep Fn from low to high to create boundary crossings
    a_g1  = torch.rand(n_g1, 4, device=device) * 2.0 - 1.0
    # Override Fn channel: full sweep [-1, 1]
    a_g1[:, 2] = torch.rand(n_g1, device=device) * 2.0 - 1.0

    # ── G1 EXTREME sub-group: cN at 90-110% with max Fn pressure ──────────────
    # Mirrors the exact validation scenario (cN 85-100%, a_t=ones) so the APN
    # learns to project aggressively in this hardest corner of the constraint.
    n_g1_extreme = n_g1 // 3
    cx_g1e = 0.5 + torch.rand(n_g1_extreme, device=device) * 5.0
    cN_g1e = N_LIMIT_PATH * (0.90 + torch.rand(n_g1_extreme, device=device) * 0.20)
    cq_g1e = torch.rand(n_g1_extreme, device=device) * cx_g1e * RATIO_LIMIT * 0.5
    V_g1e  = 20.0 + torch.rand(n_g1_extreme, device=device) * 25.0
    t_g1e  = torch.rand(n_g1_extreme, device=device) * 0.8
    # Max-pressure actions: Fn forced near maximum, other dims random
    a_g1e  = torch.rand(n_g1_extreme, 4, device=device) * 2.0 - 1.0
    a_g1e[:, 2] = 0.5 + torch.rand(n_g1_extreme, device=device) * 0.5  # Fn in [0.5, 1.0]

    # ══════════════════════════════════════════════════════════════════════════
    # G2 boundary — cq/cx near ratio limit
    # ══════════════════════════════════════════════════════════════════════════
    cx_g2 = 0.3 + torch.rand(n_g2, device=device) * 5.0
    # Product ratio from 80% to 120% of limit (tight boundary focus)
    ratio_target = RATIO_LIMIT * (0.80 + torch.rand(n_g2, device=device) * 0.40)
    cq_g2 = (cx_g2 * ratio_target).clamp(0.0, 0.2)
    # Moderate-to-high cN so biomass growth is active (cx growth dilutes ratio)
    cN_g2 = N_LIMIT_PATH * (0.25 + torch.rand(n_g2, device=device) * 0.50)
    V_g2  = 20.0 + torch.rand(n_g2, device=device) * 25.0
    t_g2  = torch.rand(n_g2, device=device) * 0.8
    a_g2  = torch.rand(n_g2, 4, device=device) * 2.0 - 1.0
    # Full light sweep: at low I, phi_Iq/phi_I ratio is higher → ratio worsens;
    # at high I, photoinhibition of pigment > growth → ratio improves.
    # APN must learn this gradient direction correctly.
    a_g2[:, 1] = torch.rand(n_g2, device=device) * 2.0 - 1.0  # full [-1, 1]

    # ══════════════════════════════════════════════════════════════════════════
    # G4 boundary — V near overflow with variable feed
    # ══════════════════════════════════════════════════════════════════════════
    cx_g4 = 0.5 + torch.rand(n_g4, device=device) * 5.0
    cN_g4 = torch.rand(n_g4, device=device) * 600.0
    cq_g4 = torch.rand(n_g4, device=device) * cx_g4 * RATIO_LIMIT * 0.6
    # V from 75% to 110% of V_MAX
    V_g4  = V_MAX * (0.75 + torch.rand(n_g4, device=device) * 0.35)
    t_g4  = torch.rand(n_g4, device=device) * 0.6
    a_g4  = torch.rand(n_g4, 4, device=device) * 2.0 - 1.0
    # Sweep Fn: high feed creates overflow
    a_g4[:, 2] = torch.rand(n_g4, device=device) * 2.0 - 1.0



    # ══════════════════════════════════════════════════════════════════════════
    # CYCLE-INITIAL samples — pre-cycle AND post-cycle partial-reset states
    # Physical profile mirrors _partial_reset_reactor(): Cx≈1.1, CN≈150,
    # Cq≈0.01, V≈V_RESET (5 L).  The reactor is refilled from V_RESET after each
    # harvest, so the starting volume for a new inoculation batch is low.
    # Split into pre-cycle (t≈0, full supply) and post-cycle (t>0.25, depleted
    # supply) temporal contexts so the APN generalises across the full episode
    # timeline.  65% G1-boundary focus (high Fn sweep → CN accumulation risk)
    # and 35% safe interior.
    # ══════════════════════════════════════════════════════════════════════════
    n_post_bnd = int(n_postcycle * 0.65)
    n_post_int = n_postcycle - n_post_bnd

    # --- G1-boundary sub-group: moderate CN with full Fn sweep ---------------
    cx_pb = 0.8 + torch.rand(n_post_bnd, device=device) * 1.7     # 0.8–2.5
    cN_pb = 100.0 + torch.rand(n_post_bnd, device=device) * 580.0 # 100–680
    cq_pb = torch.rand(n_post_bnd, device=device) * 0.04          # ≈0
    # Volume starts near V_RESET (5 L) and may have grown a little via feed
    V_pb  = V_RESET + torch.rand(n_post_bnd, device=device) * 15.0  # 5–20 L
    a_pb  = torch.rand(n_post_bnd, 4, device=device) * 2.0 - 1.0
    a_pb[:, 2] = torch.rand(n_post_bnd, device=device) * 2.0 - 1.0  # full Fn
    # Temporal split: 40% pre-cycle (t≈0), 60% post-cycle (t>0.25)
    n_pre_bnd  = int(n_post_bnd * 0.4)
    t_pb = torch.empty(n_post_bnd, device=device)
    t_pb[:n_pre_bnd]  = torch.rand(n_pre_bnd, device=device) * 0.15    # 0.0–0.15
    t_pb[n_pre_bnd:]  = 0.25 + torch.rand(n_post_bnd - n_pre_bnd, device=device) * 0.55  # 0.25–0.80

    # --- Safe interior sub-group: comfortably below all limits ---------------
    cx_pi = 0.8 + torch.rand(n_post_int, device=device) * 1.2     # 0.8–2.0
    cN_pi = 80.0 + torch.rand(n_post_int, device=device) * 200.0  # 80–280
    cq_pi = torch.rand(n_post_int, device=device) * 0.02          # ≈0
    # Volume starts near V_RESET (5 L)
    V_pi  = V_RESET + torch.rand(n_post_int, device=device) * 10.0  # 5–15 L
    a_pi  = torch.rand(n_post_int, 4, device=device) * 1.4 - 0.7  # moderate
    n_pre_int  = int(n_post_int * 0.4)
    t_pi = torch.empty(n_post_int, device=device)
    t_pi[:n_pre_int]  = torch.rand(n_pre_int, device=device) * 0.15
    t_pi[n_pre_int:]  = 0.25 + torch.rand(n_post_int - n_pre_int, device=device) * 0.55

    cx_post = torch.cat([cx_pb, cx_pi])
    cN_post = torch.cat([cN_pb, cN_pi])
    cq_post = torch.cat([cq_pb, cq_pi])
    V_post  = torch.cat([V_pb, V_pi])
    t_post  = torch.cat([t_pb, t_pi])
    a_post  = torch.cat([a_pb, a_pi])

    # ══════════════════════════════════════════════════════════════════════════
    # Concatenate all samples
    # ══════════════════════════════════════════════════════════════════════════
    cx = torch.cat([cx_int, cx_g1, cx_g1e, cx_g2, cx_g4, cx_post])
    cN = torch.cat([cN_int, cN_g1, cN_g1e, cN_g2, cN_g4, cN_post])
    cq = torch.cat([cq_int, cq_g1, cq_g1e, cq_g2, cq_g4, cq_post])
    V  = torch.cat([V_int,  V_g1,  V_g1e,  V_g2,  V_g4,  V_post]).clamp(V_MIN * 0.3, V_MAX * 1.3)
    t_norm = torch.cat([t_int, t_g1, t_g1e, t_g2, t_g4, t_post])
    actions = torch.cat([a_int, a_g1, a_g1e, a_g2, a_g4, a_post])

    num_samples = cx.shape[0]

    # ── Stage assignment ──────────────────────────────────────────────────────
    stage_idx = torch.randint(0, 4, (num_samples,), device=device)
    # G1 main samples: Inoculation (stage 0) — nitrate feed is at FN_MAX_GROWTH
    stage_idx[n_interior : n_interior + n_g1] = 0
    # G1 extreme sub-group: also Inoculation
    g1e_start = n_interior + n_g1
    stage_idx[g1e_start : g1e_start + n_g1_extreme] = 0
    # Mix some G4 samples into growth stage (stage 0) where feed is active
    g4_start = n_interior + n_g1 + n_g1_extreme + n_g2
    stage_idx[g4_start:g4_start + n_g4] = torch.where(
        torch.rand(n_g4, device=device) > 0.3,
        torch.tensor(0, device=device),
        stage_idx[g4_start:g4_start + n_g4]
    )
    # G2 samples: mostly production stage (stage 1)
    g2_start = n_interior + n_g1 + n_g1_extreme
    stage_idx[g2_start:g2_start + n_g2] = torch.where(
        torch.rand(n_g2, device=device) > 0.3,
        torch.tensor(1, device=device),
        stage_idx[g2_start:g2_start + n_g2]
    )
    # Cycle-initial samples: always Inoculation stage (just started / backtracked)
    post_start = n_interior + n_g1 + n_g1_extreme + n_g2 + n_g4
    stage_idx[post_start:post_start + n_postcycle] = 0

    stage_onehot = torch.zeros(num_samples, 4, device=device)
    stage_onehot.scatter_(1, stage_idx.unsqueeze(1), 1.0)

    # Credit remaining
    credit_norm = torch.rand(num_samples, device=device)

    # Supply remaining
    supply_norm = torch.rand(num_samples, device=device)

    # Cycle-initial overrides: fresh Inoculation credits, supply split pre/post
    n_pre_total = int(n_postcycle * 0.4)
    credit_norm[post_start:post_start + n_postcycle] = (
        0.7 + torch.rand(n_postcycle, device=device) * 0.3)   # near-full credits
    # Pre-cycle: full supply; post-cycle: partially depleted
    supply_norm[post_start:post_start + n_pre_total] = (
        0.85 + torch.rand(n_pre_total, device=device) * 0.15)  # 0.85–1.0
    supply_norm[post_start + n_pre_total:post_start + n_postcycle] = (
        0.15 + torch.rand(n_postcycle - n_pre_total, device=device) * 0.55)  # 0.15–0.70

    # Target extreme overflow boundary with a continuous sweep [-1.0, 1.0]
    # analogous to Fermentation-PPO handling of the level constraint
    extreme_overflow_slots = ((V / V_MAX) > 0.85) & (stage_onehot[:, 0].bool() | stage_onehot[:, 1].bool())
    if extreme_overflow_slots.any():
        apply_survival_overflow = extreme_overflow_slots & (
            torch.rand(num_samples, device=device) < 0.50)
        # Use full [-1.0, 1.0] sweep to map the boundary smoothly
        safe_sweep_fn = -1.0 + torch.rand(num_samples, device=device) * 2.0
        actions[:, 2] = torch.where(apply_survival_overflow, safe_sweep_fn, actions[:, 2])

    # ── Decode physical actions ───────────────────────────────────────────────
    a_scaled = (actions + 1.0) / 2.0

    # Fn depends on stage: Inoculation uses FN_MAX_GROWTH, Growth/Harvesting uses FN_MAX_PROD
    is_growth = stage_onehot[:, 0].bool()
    is_prod   = stage_onehot[:, 1].bool()
    is_cleanup = stage_onehot[:, 2].bool()
    fn_max    = torch.where(is_growth, torch.tensor(FN_MAX_GROWTH, device=device),
                torch.where(is_prod, torch.tensor(FN_MAX_PROD, device=device),
                             torch.tensor(0.0, device=device)))
    Fn_phys = a_scaled[:, 2] * fn_max

    # Mask Fn if supply is depleted
    supply_available = (supply_norm > 0.0).float()
    Fn_phys = Fn_phys * supply_available

    # Outstream: only active in Harvesting
    Fout_phys = torch.where(is_cleanup, a_scaled[:, 3] * FOUT_MAX,
                             torch.zeros(num_samples, device=device))

    # ── Forward-simulate constraints ──────────────────────────────────────────
    # Static Kinetic Parameters
    um  = 0.0572
    KN  = 393.1
    YNX = 504.5
    km  = 0.00016
    kd  = 0.281
    ks  = 178.9
    ki  = 447.1
    ksq = 23.51
    kiq = 800.0
    KNp = 16.89

    # Light intensity from action
    I_phys = I_MIN + a_scaled[:, 1] * (I_MAX - I_MIN)
    I_phys = torch.where(is_growth | is_prod, I_phys, torch.full_like(I_phys, I_MIN))
    
    # Photolimitation
    phi_I  = I_phys / (I_phys + ks + (I_phys**2 / ki))
    phi_N  = cN / (cN + KN)
    phi_Iq = I_phys / (I_phys + ksq + (I_phys**2 / kiq))

    growth = um * phi_I * cx * phi_N
    F_in_vol = Fn_phys * V / C_N_STOCK

    # Kinetics
    dCx = growth - (F_in_vol * cx / V)
    dCN = -YNX * growth + F_in_vol * (C_N_STOCK - cN) / V
    dCq = km * phi_Iq * cx - (kd * cq) / (cN + KNp) - (F_in_vol * cq / V)

    cx_next = cx + dCx * CONTROL_INTERVAL
    cN_next = cN + dCN * CONTROL_INTERVAL
    cq_next = cq + dCq * CONTROL_INTERVAL
    V_next = V + (F_in_vol - Fout_phys) * CONTROL_INTERVAL

    # G1: Nitrate path constraint
    margin_g1 = (N_LIMIT_PATH * SAFE_BUFFER - cN_next) / N_LIMIT_PATH

    # G2: Product/biomass ratio
    ratio_next = cq_next / (cx_next + 1e-8)
    margin_g2 = (RATIO_LIMIT * SAFE_BUFFER - ratio_next) / RATIO_LIMIT

    # G4: Reactor overflow
    margin_g4 = (V_MAX * SAFE_BUFFER - V_next) / V_MAX



    # ── Smooth-min across all constraints ─────────────────────────────────────
    all_margins = torch.stack([margin_g1, margin_g2, margin_g4], dim=1)
    smooth_tau  = 0.15
    min_margin  = -smooth_tau * torch.logsumexp(-all_margins / smooth_tau, dim=1)

    # ── Binary labels ─────────────────────────────────────────────────────────
    g1_vio = margin_g1 < 0.0
    g2_vio = margin_g2 < 0.0
    g4_vio = margin_g4 < 0.0
    is_unsafe = g1_vio | g2_vio | g4_vio
    is_safe   = (~is_unsafe).float()

    # Near-boundary detection
    near_g1 = (cN / N_LIMIT_PATH) > 0.70
    ratio = cq / (cx + 1e-8)
    near_g2 = ratio > (RATIO_LIMIT * 0.65)
    near_g4 = (V / V_MAX) > 0.75
    near_boundary = near_g1 | near_g2 | near_g4

    # ── Assemble normalised states (12D) ──────────────────────────────────────
    op_time_left = 1.0 - t_norm
    states = torch.cat([
        (cx / 6.0).unsqueeze(1),
        (cN / 800.0).unsqueeze(1),
        (cq / 0.2).unsqueeze(1),
        (V / V_MAX).unsqueeze(1),
        stage_onehot,
        credit_norm.unsqueeze(1),
        t_norm.unsqueeze(1),
        supply_norm.unsqueeze(1),
        op_time_left.unsqueeze(1),
    ], dim=1)  # [N, 12]

    return (states.float(), actions.float(), is_safe,
            min_margin.float(), margin_g1, margin_g2, margin_g4,
            near_boundary)


def get_fresh_batch_dataset(num_samples: int = 500000, bias: float = 0.7, pass_rates: dict = None):
    """Builds a balanced pretraining dataset by aggregating filtered raw batches.

    Guarantees a class balance of 40% safe and 60% unsafe samples. The safe
    samples are further divided into regular (interior) safe points and 
    near-boundary safe points to provide sharp gradient signal near the manifold.

    Args:
        num_samples (int, optional): Total size of the balanced dataset. Defaults to 500000.
        bias (float, optional): Passed to the raw batch generator. Defaults to 0.7.
        pass_rates (dict, optional): Passed to the raw batch generator. Defaults to None.

    Returns:
        tuple: Permuted balanced tensors (states, actions, labels, min_margin, margin_g1, margin_g2, margin_g4).
    """
    target_safe_reg = num_samples * 15 // 100
    target_safe_bnd = num_samples * 25 // 100
    target_unsafe   = num_samples - target_safe_reg - target_safe_bnd

    s_safe_reg, a_safe_reg, m_safe_reg, mg1_sr, mg2_sr, mg4_sr = [], [], [], [], [], []
    s_safe_bnd, a_safe_bnd, m_safe_bnd, mg1_sb, mg2_sb, mg4_sb = [], [], [], [], [], []
    s_unsafe, a_unsafe, m_unsafe, mg1_u, mg2_u, mg4_u = [], [], [], [], [], []

    count_safe_reg = count_safe_bnd = count_unsafe = 0

    while (count_safe_reg < target_safe_reg
           or count_safe_bnd < target_safe_bnd
           or count_unsafe   < target_unsafe):

        cand_s, cand_a, is_safe_f, cand_m, c_mg1, c_mg2, c_mg4, is_near_bnd = \
            _generate_raw_batch(num_samples, bias, pass_rates)

        is_sf = is_safe_f.bool()
        is_safe_reg_mask = is_sf & ~is_near_bnd
        is_safe_bnd_mask = is_sf &  is_near_bnd
        is_unsafe_mask   = ~is_sf

        if count_safe_reg < target_safe_reg:
            idx  = is_safe_reg_mask.nonzero(as_tuple=True)[0]
            take = min(target_safe_reg - count_safe_reg, idx.size(0))
            if take > 0:
                s_safe_reg.append(cand_s[idx[:take]])
                a_safe_reg.append(cand_a[idx[:take]])
                m_safe_reg.append(cand_m[idx[:take]])
                mg1_sr.append(c_mg1[idx[:take]])
                mg2_sr.append(c_mg2[idx[:take]])
                mg4_sr.append(c_mg4[idx[:take]])
                count_safe_reg += take

        if count_safe_bnd < target_safe_bnd:
            idx  = is_safe_bnd_mask.nonzero(as_tuple=True)[0]
            take = min(target_safe_bnd - count_safe_bnd, idx.size(0))
            if take > 0:
                s_safe_bnd.append(cand_s[idx[:take]])
                a_safe_bnd.append(cand_a[idx[:take]])
                m_safe_bnd.append(cand_m[idx[:take]])
                mg1_sb.append(c_mg1[idx[:take]])
                mg2_sb.append(c_mg2[idx[:take]])
                mg4_sb.append(c_mg4[idx[:take]])
                count_safe_bnd += take

        if count_unsafe < target_unsafe:
            idx  = is_unsafe_mask.nonzero(as_tuple=True)[0]
            take = min(target_unsafe - count_unsafe, idx.size(0))
            if take > 0:
                s_unsafe.append(cand_s[idx[:take]])
                a_unsafe.append(cand_a[idx[:take]])
                m_unsafe.append(cand_m[idx[:take]])
                mg1_u.append(c_mg1[idx[:take]])
                mg2_u.append(c_mg2[idx[:take]])
                mg4_u.append(c_mg4[idx[:take]])
                count_unsafe += take

    states  = torch.cat(s_safe_reg + s_safe_bnd + s_unsafe, dim=0)
    actions = torch.cat(a_safe_reg + a_safe_bnd + a_unsafe, dim=0)
    margins = torch.cat(m_safe_reg + m_safe_bnd + m_unsafe, dim=0)
    margins_g1 = torch.cat(mg1_sr + mg1_sb + mg1_u, dim=0)
    margins_g2 = torch.cat(mg2_sr + mg2_sb + mg2_u, dim=0)
    margins_g4 = torch.cat(mg4_sr + mg4_sb + mg4_u, dim=0)

    n_safe = target_safe_reg + target_safe_bnd
    labels = torch.full((states.size(0),), 0.05, device=states.device)
    labels[:n_safe] = 0.95

    perm = torch.randperm(states.size(0), device=states.device)
    return (states[perm], actions[perm], labels[perm], margins[perm],
            margins_g1[perm], margins_g2[perm], margins_g4[perm])