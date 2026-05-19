"""
Dataset generation for the Safety Boundary Classifier (Photoproduction).

Generates (state, action, margin) tuples covering 4 instantaneous constraints:
  G1 — Nitrate path:       cN + Fn*10h  ≤  N_LIMIT_PATH * SAFE_BUFFER
  G2 — Product ratio:      cq / cx       ≤  RATIO_LIMIT * SAFE_BUFFER
  G4 — Reactor overflow:   V + dV*10h    ≤  V_MAX * SAFE_BUFFER
  G5 — Reactor underflow:  V - dV*10h    ≥  V_MIN / SAFE_BUFFER

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
C_N_STOCK       = 50000.0
CONTROL_INTERVAL = 10.0
TOTAL_TIME      = 500.0
SAFE_BUFFER     = 0.98


def _generate_raw_batch(num_samples: int, bias: float = 0.7):
    """
    Generate a raw batch of (state, action, is_safe, margin) samples.

    State layout (11D, normalised):
      [cx/6, cN/800, cq/0.2, V/V_MAX, stage_0..3, credit/base, t/500, supply/initial]

    Action layout (4D, [-1,1]):
      [time_mult, I_norm, Fn_norm, Fout_norm]

    Sampling strategy:
      - 30% uniform interior (safe identity region)
      - 70% biased toward constraint boundaries (mixed safe/unsafe)
        - Sub-biases for each of G1, G2, G4, G5 near-boundary regions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_interior = int(num_samples * (1.0 - bias))
    n_boundary = num_samples - n_interior

    # Split boundary budget across constraints
    n_g2 = int(n_boundary * 0.4)
    n_g1 = int(n_boundary * 0.2)
    n_g4 = int(n_boundary * 0.2)
    n_g5 = n_boundary - n_g1 - n_g2 - n_g4

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
    # cN from 60% to 105% of limit — straddle the boundary
    cN_g1 = N_LIMIT_PATH * (0.60 + torch.rand(n_g1, device=device) * 0.45)
    cq_g1 = torch.rand(n_g1, device=device) * cx_g1 * RATIO_LIMIT * 0.5
    V_g1  = 20.0 + torch.rand(n_g1, device=device) * 25.0
    t_g1  = torch.rand(n_g1, device=device) * 0.8
    # Actions: sweep Fn from low to high to create boundary crossings
    a_g1  = torch.rand(n_g1, 4, device=device) * 2.0 - 1.0
    # Override Fn channel: full sweep [-1, 1]
    a_g1[:, 2] = torch.rand(n_g1, device=device) * 2.0 - 1.0

    # ══════════════════════════════════════════════════════════════════════════
    # G2 boundary — cq/cx near ratio limit
    # ══════════════════════════════════════════════════════════════════════════
    cx_g2 = 0.3 + torch.rand(n_g2, device=device) * 5.0
    # Product ratio from 70% to 130% of limit
    ratio_target = RATIO_LIMIT * (0.70 + torch.rand(n_g2, device=device) * 0.60)
    cq_g2 = (cx_g2 * ratio_target).clamp(0.0, 0.2)
    # Moderate-to-high cN so biomass growth is active (cx growth dilutes ratio)
    cN_g2 = N_LIMIT_PATH * (0.25 + torch.rand(n_g2, device=device) * 0.50)
    V_g2  = 20.0 + torch.rand(n_g2, device=device) * 25.0
    t_g2  = torch.rand(n_g2, device=device) * 0.8
    a_g2  = torch.rand(n_g2, 4, device=device) * 2.0 - 1.0
    # Bias light high: high light drives faster cx growth, diluting cq/cx ratio
    a_g2[:, 1] = torch.rand(n_g2, device=device) * 1.5 - 0.5  # [-0.5, 1.0]

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
    # G5 boundary — V near underflow with variable outflow
    # ══════════════════════════════════════════════════════════════════════════
    cx_g5 = 0.5 + torch.rand(n_g5, device=device) * 5.0
    cN_g5 = torch.rand(n_g5, device=device) * 300.0
    cq_g5 = torch.rand(n_g5, device=device) * cx_g5 * RATIO_LIMIT * 0.6
    # V from 50% of V_MIN to 300% of V_MIN (2.5 to 15 L)
    V_g5  = V_MIN * (0.50 + torch.rand(n_g5, device=device) * 2.50)
    t_g5  = 0.4 + torch.rand(n_g5, device=device) * 0.5
    a_g5  = torch.rand(n_g5, 4, device=device) * 2.0 - 1.0
    # Sweep Fout: high outflow creates underflow
    a_g5[:, 3] = torch.rand(n_g5, device=device) * 2.0 - 1.0

    # ══════════════════════════════════════════════════════════════════════════
    # Concatenate all samples
    # ══════════════════════════════════════════════════════════════════════════
    cx = torch.cat([cx_int, cx_g1, cx_g2, cx_g4, cx_g5])
    cN = torch.cat([cN_int, cN_g1, cN_g2, cN_g4, cN_g5])
    cq = torch.cat([cq_int, cq_g1, cq_g2, cq_g4, cq_g5])
    V  = torch.cat([V_int,  V_g1,  V_g2,  V_g4,  V_g5]).clamp(V_MIN * 0.3, V_MAX * 1.3)
    t_norm = torch.cat([t_int, t_g1, t_g2, t_g4, t_g5])
    actions = torch.cat([a_int, a_g1, a_g2, a_g4, a_g5])

    # ── Stage assignment ──────────────────────────────────────────────────────
    # Bias: G5 samples → cleanup stage (where outflow is active)
    stage_idx = torch.randint(0, 4, (num_samples,), device=device)
    # Force G5 samples into cleanup stage (stage 2)
    g5_start = n_interior + n_g1 + n_g2 + n_g4
    stage_idx[g5_start:] = 2
    # Mix some G4 samples into growth stage (stage 0) where feed is active
    g4_start = n_interior + n_g1 + n_g2
    stage_idx[g4_start:g4_start + n_g4] = torch.where(
        torch.rand(n_g4, device=device) > 0.3,
        torch.tensor(0, device=device),
        stage_idx[g4_start:g4_start + n_g4]
    )
    # G2 samples: mostly production stage (stage 1)
    g2_start = n_interior + n_g1
    stage_idx[g2_start:g2_start + n_g2] = torch.where(
        torch.rand(n_g2, device=device) > 0.3,
        torch.tensor(1, device=device),
        stage_idx[g2_start:g2_start + n_g2]
    )

    stage_onehot = torch.zeros(num_samples, 4, device=device)
    stage_onehot.scatter_(1, stage_idx.unsqueeze(1), 1.0)

    # Credit remaining
    credit_norm = torch.rand(num_samples, device=device)

    # Supply remaining
    supply_norm = torch.rand(num_samples, device=device)

    # ── Decode physical actions ───────────────────────────────────────────────
    a_scaled = (actions + 1.0) / 2.0

    # Fn depends on stage: Growth uses FN_MAX_GROWTH, Production uses FN_MAX_PROD
    is_growth = stage_onehot[:, 0].bool()
    is_prod   = stage_onehot[:, 1].bool()
    fn_max    = torch.where(is_growth, torch.tensor(FN_MAX_GROWTH, device=device),
                torch.where(is_prod, torch.tensor(FN_MAX_PROD, device=device),
                             torch.tensor(0.0, device=device)))
    Fn_phys = a_scaled[:, 2] * fn_max

    # Outstream: only active in Cleanup
    is_cleanup = stage_onehot[:, 2].bool()
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

    # G5: Reactor underflow
    margin_g5 = (V_next - V_MIN / SAFE_BUFFER) / V_MAX

    # ── Smooth-min across all constraints ─────────────────────────────────────
    all_margins = torch.stack([margin_g1, margin_g2,
                                margin_g4, margin_g5], dim=1)
    smooth_tau  = 0.15
    min_margin  = -smooth_tau * torch.logsumexp(-all_margins / smooth_tau, dim=1)

    # ── Binary labels ─────────────────────────────────────────────────────────
    g1_vio = margin_g1 < 0.0
    g2_vio = margin_g2 < 0.0
    g4_vio = margin_g4 < 0.0
    g5_vio = margin_g5 < 0.0
    is_unsafe = g1_vio | g2_vio | g4_vio | g5_vio
    is_safe   = (~is_unsafe).float()

    # Near-boundary detection
    near_g1 = (cN / N_LIMIT_PATH) > 0.70
    ratio = cq / (cx + 1e-8)
    near_g2 = ratio > (RATIO_LIMIT * 0.65)
    near_g4 = (V / V_MAX) > 0.75
    near_g5 = V < (V_MIN * 2.5)
    near_boundary = near_g1 | near_g2 | near_g4 | near_g5

    # ── Assemble normalised states (11D) ──────────────────────────────────────
    states = torch.cat([
        (cx / 6.0).unsqueeze(1),
        (cN / 800.0).unsqueeze(1),
        (cq / 0.2).unsqueeze(1),
        (V / V_MAX).unsqueeze(1),
        stage_onehot,
        credit_norm.unsqueeze(1),
        t_norm.unsqueeze(1),
        supply_norm.unsqueeze(1),
    ], dim=1)  # [N, 11]

    return (states.float(), actions.float(), is_safe,
            min_margin.float(), g1_vio, g2_vio, near_boundary)


def get_fresh_batch_dataset(num_samples: int = 500000, bias: float = 0.7):
    """
    Build one balanced training batch (40% safe / 60% unsafe).
    """
    target_safe_reg = num_samples * 15 // 100
    target_safe_bnd = num_samples * 25 // 100
    target_unsafe   = num_samples - target_safe_reg - target_safe_bnd

    s_safe_reg, a_safe_reg, m_safe_reg = [], [], []
    s_safe_bnd, a_safe_bnd, m_safe_bnd = [], [], []
    s_unsafe,   a_unsafe,   m_unsafe   = [], [], []

    count_safe_reg = count_safe_bnd = count_unsafe = 0

    while (count_safe_reg < target_safe_reg
           or count_safe_bnd < target_safe_bnd
           or count_unsafe   < target_unsafe):

        cand_s, cand_a, is_safe_f, cand_m, _g1, _g2, is_near_bnd = \
            _generate_raw_batch(num_samples, bias)

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
                count_safe_reg += take

        if count_safe_bnd < target_safe_bnd:
            idx  = is_safe_bnd_mask.nonzero(as_tuple=True)[0]
            take = min(target_safe_bnd - count_safe_bnd, idx.size(0))
            if take > 0:
                s_safe_bnd.append(cand_s[idx[:take]])
                a_safe_bnd.append(cand_a[idx[:take]])
                m_safe_bnd.append(cand_m[idx[:take]])
                count_safe_bnd += take

        if count_unsafe < target_unsafe:
            idx  = is_unsafe_mask.nonzero(as_tuple=True)[0]
            take = min(target_unsafe - count_unsafe, idx.size(0))
            if take > 0:
                s_unsafe.append(cand_s[idx[:take]])
                a_unsafe.append(cand_a[idx[:take]])
                m_unsafe.append(cand_m[idx[:take]])
                count_unsafe += take

    states  = torch.cat(s_safe_reg + s_safe_bnd + s_unsafe, dim=0)
    actions = torch.cat(a_safe_reg + a_safe_bnd + a_unsafe, dim=0)
    margins = torch.cat(m_safe_reg + m_safe_bnd + m_unsafe, dim=0)

    n_safe = target_safe_reg + target_safe_bnd
    labels = torch.full((states.size(0),), 0.05, device=states.device)
    labels[:n_safe] = 0.95

    perm = torch.randperm(states.size(0), device=states.device)
    return states[perm], actions[perm], labels[perm], margins[perm]