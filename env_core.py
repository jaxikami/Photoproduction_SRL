import numpy as np
from numba import njit

# =============================================================================
# KINETIC ENGINE (Photoproduction CSTR with Volume Tracking)
# =============================================================================

# Feed stock concentration (mg/L) — concentrated nitrate solution
C_N_STOCK = 50000.0

@njit
def calculate_rates_numba(state, I, Fn, F_out):
    """
    Computes instantaneous rates for the bioreactor state [Cx, CN, Cq, V].

    The CSTR formulation accounts for dilution effects from feed inflow
    and product removal via outstream.

    Args:
        state (array): [Biomass Cx (g/L), Nitrate CN (mg/L),
                        Phycocyanin Cq (mg/L), Volume V (L)].
        I (float):     Light intensity (umol/m^2/s).
        Fn (float):    Nitrate feed concentration rate (mg/L/h).
        F_out (float): Outstream volumetric flow (L/h).

    Returns:
        R (array): [dCx/dt, dCN/dt, dCq/dt, dV/dt].
    """
    Cx = state[0]
    CN = state[1]
    Cq = state[2]
    V  = max(state[3], 0.1)

    # Static Kinetic Parameters
    um  = 0.0572     # h^-1
    ud  = 0.0        # h^-1 (Death rate)
    KN  = 393.1      # mg/L
    YNX = 504.5      # mg/g
    km  = 0.00016    # mg/g/h
    kd  = 0.281      # h^-1
    ks  = 178.9      # umol/m^2/s
    ki  = 447.1      # umol/m^2/s
    ksq = 23.51      # umol/m^2/s
    kiq = 800.0      # umol/m^2/s
    KNp = 16.89      # mg/L

    # Volumetric inflow from concentrated nitrate feed
    F_in_vol = max(0.0, Fn * V / C_N_STOCK)   # L/h

    # Growth photolimitation (Aiba model)
    phi_I  = I / (I + ks + (I**2 / ki))
    # Nutrient availability
    phi_N  = CN / (CN + KN)
    # Product photolimitation
    phi_Iq = I / (I + ksq + (I**2 / kiq))

    growth = um * phi_I * Cx * phi_N

    R = np.zeros(4)
    # dCx/dt: growth – death – dilution from inflow (no biomass in feed)
    R[0] = growth - ud * Cx - F_in_vol * Cx / V
    # dCN/dt: consumption + CSTR feed term
    R[1] = -YNX * growth + F_in_vol * (C_N_STOCK - CN) / V
    # dCq/dt: synthesis – degradation – dilution from inflow
    R[2] = km * phi_Iq * Cx - (kd * Cq) / (CN + KNp) - F_in_vol * Cq / V
    # dV/dt: inflow – outflow
    R[3] = F_in_vol - F_out

    return R


@njit
def integrate_rk4(state_init, I, Fn, F_out, dt, n_steps):
    """
    4th-order Runge-Kutta integrator for the CSTR system [Cx, CN, Cq, V].
    """
    state = state_init.copy()
    for _ in range(n_steps):
        k1 = calculate_rates_numba(state, I, Fn, F_out)
        k2 = calculate_rates_numba(state + 0.5 * dt * k1, I, Fn, F_out)
        k3 = calculate_rates_numba(state + 0.5 * dt * k2, I, Fn, F_out)
        k4 = calculate_rates_numba(state + dt * k3, I, Fn, F_out)
        state += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        # Clamp non-physical values
        state[0] = max(state[0], 0.0)
        state[1] = max(state[1], 0.0)
        state[2] = max(state[2], 0.0)
        state[3] = max(state[3], 0.1)
    return state


# =============================================================================
# REINFORCEMENT LEARNING ENVIRONMENT
# =============================================================================

class PhycocyaninEnvCore:
    """
    Multi-stage photobioreactor RL environment core.

    Stages:
        0 — Growth:     Rapid biomass accumulation (I: 120-400, Fn: 0-40)
        1 — Production: Bioproduct synthesis      (I: 120-400, Fn: 0-10)
        2 — Cleanup:    Reactor harvest/drain      (Fout active, I/Fn masked)
                        → stage 3 (volume latch) OR stage 0 (credit expiry, partial reset)
        3 — Idle:       Reactor off, system shutdown (all masked)

    Action dimensions (4):
        0 — Time multiplier  (0.5–2.0, active in Growth/Production)
        1 — Light intensity  (120–400 μmol/m²/s, active in Growth/Production)
        2 — Nitrate feed     (0–40 or 0–10 mg/L/h, active in Growth/Production)
        3 — Outstream flow   (0–Fout_max L/h, active in Cleanup only)
    """

    def __init__(self):
        # --- Simulation Config ---
        self.total_time = 500.0
        self.control_freq = 10.0
        self.max_steps = int(self.total_time / self.control_freq)  # 50

        # --- Stage Config ---
        self.n_stage = 4
        self.BASE_CREDITS = np.array([120.0, 60.0, 50.0, 0.0])
        #                              Growth  Prod  Cleanup  Idle

        # --- Physical Action Boundaries ---
        self.I_MIN, self.I_MAX = 120.0, 400.0
        self.FN_MAX_GROWTH = 40.0
        self.FN_MAX_PROD   = 10.0
        self.FOUT_MAX      = 2.0   # L/h max outstream

        # --- Reactor Volume Specs ---
        self.V_MAX     = 50.0    # L — reactor capacity
        self.V_MIN     = 5.0     # L — dry floor
        self.V_INITIAL = 40.0    # L — initial fill (80%)
        self.V_DRAIN   = 10.0    # L — cleanup→idle trigger

        # --- Global Nutrient Pool ---
        self.INITIAL_NITRATE_SUPPLY = 250_000.0  # mg total

        # --- Constraint Limits ---
        self.N_LIMIT_PATH  = 800.0    # g1: Max path nitrate (mg/L)
        self.RATIO_LIMIT   = 0.011    # g2: Max cq/cx ratio
        self.N_LIMIT_TERM  = 150.0    # g3: Terminal nitrate (mg/L)

        # --- Buffer / Barrier Zone Config ---
        self.OVERFLOW_BUFFER_FRAC = 0.10   # g4 buffer activates at 90% V_MAX

        # --- Integration Config ---
        self.dt = 10.0 / 60.0   # 10 minutes (0.1667 h)
        self.n_inner_steps = int(self.control_freq / self.dt)  # 60

        self.reset()

    def reset(self, randomize=False):
        """
        Resets the environment for a new training episode.
        """
        self.time = 0.0
        self.time_step_count = 0

        # Stage scheduling
        self.current_stage = 0
        self.stage_credits = float(self.BASE_CREDITS[0])

        # Hysteretic latch for cleanup→idle transition
        self._cleanup_latch = False

        # Physical state: [Cx (g/L), CN (mg/L), Cq (mg/L), V (L)]
        self.state = np.array([1.1, 150.0, 0.01, self.V_INITIAL], dtype=np.float64)

        if randomize:
            noise_factor = 0.10
            # Only randomize concentrations, not volume
            conc_noise = np.random.normal(1.0, noise_factor, size=3)
            self.state[:3] *= np.maximum(0.01, conc_noise)
            # Slight volume randomization
            self.state[3] = np.clip(
                self.state[3] * np.random.normal(1.0, 0.05),
                self.V_MIN + 5.0, self.V_MAX - 5.0
            )

        # Global nutrient pool
        self.nitrate_supply = self.INITIAL_NITRATE_SUPPLY

        # Action smoothing
        self.prev_action = np.zeros(4)

        # Violation tracking
        self.violation_count = 0
        self.g1_violation_count = 0
        self.g2_violation_count = 0
        self.g3_violation_count = 0
        self.g4_violation_count = 0
        self.g5_violation_count = 0

        # Metric tracking
        self.ep_total_reward = 0.0
        self.ep_rewards = []
        self.ep_prod_rewards = []
        self.ep_smooth_penalties = []
        self.ep_constraint_penalties = []
        self.ep_raw_mat_penalties = []
        self.ep_g1_penalties = []
        self.ep_g2_penalties = []
        self.ep_g3_penalties = []
        self.ep_g4_penalties = []
        self.ep_g5_penalties = []

        # Phycocyanin harvested during cleanup (mass in mg)
        self.total_cq_harvested = 0.0

        return self.get_state_norm()

    def get_state_norm(self):
        """
        Normalized observation vector (12D):
          [0] Cx/6.0, [1] CN/800.0, [2] Cq/0.2, [3] V/V_max,
          [4-7] stage one-hot, [8] credit_remaining/base,
          [9] t/500, [10] supply_remaining/initial,
          [11] operation_time_left (1 - t/500)
        """
        norm = np.zeros(12, dtype=np.float64)
        norm[0] = self.state[0] / 6.0
        norm[1] = self.state[1] / 800.0
        norm[2] = self.state[2] / 0.2
        norm[3] = self.state[3] / self.V_MAX

        # One-hot stage encoding
        norm[4 + self.current_stage] = 1.0

        # Credit remaining (normalized to base credit of current stage)
        base = self.BASE_CREDITS[self.current_stage]
        norm[8] = (self.stage_credits / base) if base > 0 else 0.0

        norm[9]  = self.time / self.total_time
        norm[10] = self.nitrate_supply / self.INITIAL_NITRATE_SUPPLY
        norm[11] = 1.0 - self.time / self.total_time  # operation time left

        return norm

    @staticmethod
    def get_action_mask(state_tensor):
        """
        Computes the stage-aware binary action mask from the state tensor.

        Returns mask of shape [..., 4]:
          dim 0 — time multiplier:  active in Growth (0) & Production (1)
          dim 1 — light intensity:  active in Growth (0) & Production (1)
          dim 2 — nitrate feed:     active in Growth (0) & Production (1)
          dim 3 — outstream flow:   active in Cleanup (2) only
        """
        import torch
        # Stage one-hot at indices 4:8
        stage_0 = state_tensor[..., 4]   # Growth
        stage_1 = state_tensor[..., 5]   # Production
        stage_2 = state_tensor[..., 6]   # Cleanup
        # stage_3 = state_tensor[..., 7]  # Idle — all masked

        mask_time = stage_0 + stage_1
        mask_I    = stage_0 + stage_1
        mask_Fn   = stage_0 + stage_1
        mask_Fout = stage_2

        return torch.stack([mask_time, mask_I, mask_Fn, mask_Fout], dim=-1)

    def _sigmoid_switch(self, x, center=0.0, width=2.0):
        """Smooth transition factor ∈ [0, 1] for hysteretic switching."""
        return 1.0 / (1.0 + np.exp(-(x - center) / max(width, 0.01)))

    def _partial_reset_reactor(self):
        """
        Reset the physical reactor state for a new Growth batch.
        Preserves episode tracking (time, step count, violations, rewards).
        Called when Cleanup credits expire → stage 0 backtrack.
        """
        self.state = np.array([1.1, 150.0, 0.01, self.V_INITIAL], dtype=np.float64)
        self.nitrate_supply = self.INITIAL_NITRATE_SUPPLY
        self.prev_action = np.zeros(4)


    def _physics_step(self, action):
        """
        Executes one control step (10 hours) physics simulation.
        Returns:
            a_clipped (ndarray): The clipped action.
            Fn_phys (float): The actual nitrate feed applied.
            done (bool): Whether the max time steps has been reached.
        """
        a_clipped = np.clip(action, -1.0, 1.0)
        a_scaled  = (a_clipped + 1.0) / 2.0   # [0, 1]

        # ── Action Decoding (stage-dependent) ─────────────────────────
        if self.current_stage in (0, 1):
            # Time multiplier: 0.5–2.0
            multiplier = 0.5 + a_scaled[0] * 1.5

            # Light intensity
            I_phys = self.I_MIN + a_scaled[1] * (self.I_MAX - self.I_MIN)

            # Nitrate feed (stage-dependent cap)
            fn_max = self.FN_MAX_GROWTH if self.current_stage == 0 else self.FN_MAX_PROD
            Fn_phys = a_scaled[2] * fn_max

            # Outstream masked
            F_out = 0.0
        elif self.current_stage == 2:
            # Cleanup: only outstream active
            multiplier = 1.0
            I_phys  = self.I_MIN  # baseline
            Fn_phys = 0.0
            F_out   = a_scaled[3] * self.FOUT_MAX
        else:
            # Idle: everything at baseline
            multiplier = 1.0
            I_phys  = self.I_MIN
            Fn_phys = 0.0
            F_out   = 0.0

        # ── Mass Balance: Cap Fn by available supply ──────────────────
        if Fn_phys > 0 and self.nitrate_supply > 0:
            # Approximate demand over control period
            nitrate_demand = Fn_phys * self.state[3] * self.control_freq  # mg
            actual_consumed = min(nitrate_demand, self.nitrate_supply)
            if actual_consumed < nitrate_demand and nitrate_demand > 0:
                Fn_phys *= (actual_consumed / nitrate_demand)
            self.nitrate_supply -= actual_consumed
        elif self.nitrate_supply <= 0:
            Fn_phys = 0.0

        # ── Hysteretic Sigmoid Blending (Cleanup → Idle) ─────────────
        blend = 0.0
        if self.current_stage == 2:
            if not self._cleanup_latch and self.state[3] < self.V_DRAIN:
                self._cleanup_latch = True

            if self._cleanup_latch:
                # Sigmoid soft-switch: blend towards idle as V drops
                blend = self._sigmoid_switch(
                    self.V_DRAIN - self.state[3], center=0.0, width=2.0)
                # Blend actions toward idle defaults
                I_phys  = I_phys * (1.0 - blend) + self.I_MIN * blend
                Fn_phys = Fn_phys * (1.0 - blend)
                F_out   = F_out * (1.0 - blend)

        # ── Time Credit Update ────────────────────────────────────────
        dt_hours = self.control_freq   # effective step duration
        if self.current_stage in (0, 1):
            self.stage_credits -= multiplier * dt_hours
        elif self.current_stage == 2:
            self.stage_credits -= dt_hours  # fixed rate in cleanup

        # ── Stage Transition Logic ────────────────────────────────────
        if self.current_stage in (0, 1) and self.stage_credits <= 0:
            self.current_stage += 1
            self.stage_credits = float(self.BASE_CREDITS[self.current_stage])
            self._cleanup_latch = False

        elif self.current_stage == 2:
            # Immediate transition bypass if harvest is complete
            if self._cleanup_latch and blend >= 0.95:
                self.stage_credits = 0.0

            # End of stage 2 is reached when stage_credits <= 0 AND harvest is complete (blend >= 0.95)
            # If credits expire but harvest is not complete, hold stage 2 (like fermentation env)
            if self.stage_credits <= 0:
                if not (self._cleanup_latch and blend >= 0.95):
                    self.stage_credits = 0.0
                else:
                    # End of stage 2 reached!
                    min_time_for_cycle = 110.0
                    time_remaining = self.total_time - self.time
                    
                    if time_remaining >= min_time_for_cycle:
                        # Backtrack to Growth (stage 0) to start a new batch
                        self._partial_reset_reactor()
                        self.current_stage = 0
                        self.stage_credits = float(self.BASE_CREDITS[0])
                        self._cleanup_latch = False
                    else:
                        # Transition to Idle (stage 3)
                        self.current_stage = 3
                        self.stage_credits = 0.0


        # ── CSTR Integration ─────────────────────────────────────────
        self.state = integrate_rk4(
            self.state, I_phys, Fn_phys, F_out,
            self.dt, self.n_inner_steps)

        # Track harvested phycocyanin mass during cleanup outflow
        if F_out > 0:
            self.total_cq_harvested += self.state[2] * F_out * self.control_freq

        self.time += self.control_freq
        self.time_step_count += 1

        done = self.time_step_count >= self.max_steps

        return a_clipped, Fn_phys, done