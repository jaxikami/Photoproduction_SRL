"""
Core Environment Module for the Photoproduction Bioreactor.

This module provides the physics simulation engine and the base environment
class `PhycocyaninEnvCore` which handles the mass balance equations,
Runge-Kutta integration, and stage transitions of the photobioreactor
for phycocyanin production.

Volume-less formulation: concentrations evolve by pure kinetics with no
dilution terms. Fn enters dCN/dt directly as mg/L/h. state[3] is a derived
total mass concentration (mg/L) = Cx*1000 + CN + Cq.
"""
import numpy as np
from numba import njit

# =============================================================================
# KINETIC ENGINE (Volume-Less Photoproduction)
# =============================================================================

@njit
def calculate_rates_numba(state, I, Fn):
    """Computes instantaneous kinetic rates for the bioreactor state.

    Volume-less formulation: concentrations evolve by pure kinetics.
    Fn enters dCN/dt directly as a concentration addition rate (mg/L/h).
    No dilution terms. state[3] (total mass conc.) is derived, not integrated.

    Args:
        state (np.ndarray): Current physical state [Biomass Cx (g/L),
            Nitrate CN (mg/L), Phycocyanin Cq (mg/L), M_total (mg/L)].
        I (float): Light intensity (umol/m^2/s).
        Fn (float): Nitrate feed concentration rate (mg/L/h).

    Returns:
        np.ndarray: Array of instantaneous rates of change
            [dCx/dt, dCN/dt, dCq/dt, 0.0].
    """
    Cx = state[0]
    CN = state[1]
    Cq = state[2]

    # Static Kinetic Parameters
    um  = 0.0572     # h^-1
    ud  = 0.0        # h^-1 (Death rate)
    KN  = 393.1      # mg/L
    YNX = 504.5      # mg/g
    km  = 0.00016    # mg/g/h
    kd  = 0.281      # mg/L/h
    ks  = 178.9      # umol/m^2/s
    ki  = 447.1      # umol/m^2/s
    ksq = 23.51      # umol/m^2/s
    kiq = 800.0      # umol/m^2/s
    KNp = 16.89      # mg/L

    # Growth photolimitation (Aiba model)
    phi_I  = I / (I + ks + (I**2 / ki))
    # Nutrient availability
    phi_N  = CN / (CN + KN)
    # Product photolimitation
    phi_Iq = I / (I + ksq + (I**2 / kiq))

    growth = um * phi_I * Cx * phi_N

    R = np.zeros(4)
    # dCx/dt: growth – death (no dilution)
    R[0] = growth - ud * Cx
    # dCN/dt: consumption + direct feed addition (no dilution)
    R[1] = -YNX * growth + Fn
    # dCq/dt: synthesis – degradation (no dilution)
    R[2] = km * phi_Iq * Cx - (kd * Cq) / (CN + KNp)
    # state[3] is derived (M_total = Cx*1000 + CN + Cq), not integrated
    R[3] = 0.0

    return R


@njit
def integrate_rk4(state_init, I, Fn, dt, n_steps):
    """Executes a 4th-order Runge-Kutta integration for the volume-less system.

    Advances the physical state [Cx, CN, Cq] using pure kinetics.
    state[3] (total mass concentration) is recomputed after integration.

    Args:
        state_init (np.ndarray): The initial physical state before integration.
        I (float): Light intensity applied during the step.
        Fn (float): Nitrate feed rate applied during the step (mg/L/h).
        dt (float): Inner integration time step duration (hours).
        n_steps (int): Number of integration steps to execute.

    Returns:
        np.ndarray: The updated physical state after integration.
    """
    state = state_init.copy()
    for _ in range(n_steps):
        k1 = calculate_rates_numba(state, I, Fn)
        k2 = calculate_rates_numba(state + 0.5 * dt * k1, I, Fn)
        k3 = calculate_rates_numba(state + 0.5 * dt * k2, I, Fn)
        k4 = calculate_rates_numba(state + dt * k3, I, Fn)
        state += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        # Clamp non-physical values
        state[0] = max(state[0], 0.0)
        state[1] = max(state[1], 0.0)
        state[2] = max(state[2], 0.0)
    # Recompute derived total mass concentration (mg/L)
    state[3] = state[0] * 1000.0 + state[1] + state[2]
    return state


# =============================================================================
# REINFORCEMENT LEARNING ENVIRONMENT
# =============================================================================

class PhycocyaninEnvCore:
    """
    Multi-stage photobioreactor RL environment core.

    Stages:
        0 — Inoculation: Rapid biomass accumulation (I: 120-400, Fn: 0-40)
        1 — Growth:       Bioproduct synthesis      (I: 120-400, Fn: 0-10)
        2 — Harvesting:   Reactor harvest/drain      (Fout active, I/Fn masked)
                        → stage 3 (volume latch) OR stage 0 (credit expiry, partial reset)
        3 — Idle:         Reactor off, system shutdown (all masked)

    Action dimensions (4):
        0 — Time multiplier  (0.5–2.0, active in Inoculation/Growth)
        1 — Light intensity  (120–400 μmol/m²/s, active in Inoculation/Growth)
        2 — Nitrate feed     (0–40 or 0–10 mg/L/h, active in Inoculation/Growth)
        3 — Outstream flow   (0–Fout_max L/h, active in Harvesting only)
    """

    def __init__(self):
        # --- Simulation Config ---
        self.total_time = 1000.0
        self.control_freq = 10.0
        self.max_steps = int(self.total_time / self.control_freq)  # 100

        # --- Stage Config ---
        self.n_stage = 4
        self.BASE_CREDITS = np.array([163.0, 81.0, 68.0, 0.0])
        #                              Inoc   Growth  Harvest  Idle

        # --- Physical Action Boundaries ---
        self.I_MIN, self.I_MAX = 120.0, 400.0
        self.FN_MAX_GROWTH = 40.0
        self.FN_MAX_PROD   = 10.0
        self.FOUT_MAX      = 0.05  # h^-1 fractional drain rate

        # --- Total Mass Concentration Constraint (g4) ---
        self.M_CONC_LIMIT  = 5000.0  # mg/L — total dissolved concentration limit
        self.DRAIN_FRAC    = 0.25    # remaining fraction to trigger idle

        # --- Global Nutrient Pool (per-episode concentration budget) ---
        self.INITIAL_NITRATE_SUPPLY = 8000.0  # mg/L concentration budget

        # --- Constraint Limits ---
        self.N_LIMIT_PATH  = 800.0    # g1: Max path nitrate (mg/L)
        self.RATIO_LIMIT   = 0.011    # g2: Max cq/cx ratio
        self.N_LIMIT_TERM  = 150.0    # g3: Terminal nitrate (mg/L)

        # --- Buffer / Barrier Zone Config ---
        self.OVERFLOW_BUFFER_FRAC = 0.10   # g4 buffer activates at 90% of limit

        # --- Integration Config ---
        self.dt = 10.0 / 60.0   # 10 minutes (0.1667 h)
        self.n_inner_steps = int(self.control_freq / self.dt)  # 60

        self.reset()

    def reset(self, randomize=False):
        """Resets the environment for a new training episode.

        Initializes all tracking variables, physical states, and global nutrient
        pools. If `randomize` is True, initial concentrations are slightly perturbed.

        Args:
            randomize (bool, optional): Whether to inject initial state noise.
                Defaults to False.

        Returns:
            np.ndarray: The normalized initial observation vector.
        """
        self.time = 0.0
        self.time_step_count = 0

        # Stage scheduling
        self.current_stage = 0
        self.stage_credits = float(self.BASE_CREDITS[0])

        # Hysteretic latch for cleanup→idle transition
        self._cleanup_latch = False

        # Drain progress fraction (1.0 = full, 0.0 = empty)
        self.remaining_frac = 1.0

        # Physical state: [Cx (g/L), CN (mg/L), Cq (mg/L), M_total (mg/L)]
        self.state = np.array([1.1, 150.0, 0.005, 0.0], dtype=np.float64)
        self.state[3] = self.state[0] * 1000.0 + self.state[1] + self.state[2]

        if randomize:
            noise_factor = 0.10
            # Randomize concentrations
            conc_noise = np.random.normal(1.0, noise_factor, size=3)
            self.state[:3] *= np.maximum(0.01, conc_noise)
            # Recompute derived total mass concentration
            self.state[3] = self.state[0] * 1000.0 + self.state[1] + self.state[2]

        # Global nutrient pool (per-episode concentration budget, mg/L)
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
        self.ep_harvest_rewards = []
        self.ep_smooth_penalties = []
        self.ep_constraint_penalties = []
        self.ep_raw_mat_penalties = []
        self.ep_g1_penalties = []
        self.ep_g2_penalties = []
        self.ep_g3_penalties = []
        self.ep_g4_penalties = []
        self.ep_g5_penalties = []

        # Phycocyanin harvested during cleanup (concentration equivalent, mg/L)
        self.total_cq_harvested = 0.0

        return self.get_state_norm()

    def get_state_norm(self):
        """Calculates and returns the normalized observation vector.

        The observation vector provides the RL agent with a standardized scale
        (typically [0, 1] bounds) for all features, improving network training.

        Returns:
            np.ndarray: A 12-dimensional normalized vector containing:
                - [0] Cx/6.0
                - [1] CN/800.0
                - [2] Cq/0.2
                - [3] M_total/M_CONC_LIMIT (total mass concentration)
                - [4-7] One-hot encoded current stage (0 to 3)
                - [8] Remaining stage credit (normalized by base credit)
                - [9] Normalized episode time (t / total_time)
                - [10] Normalized remaining nitrate supply
                - [11] Remaining operation time fraction (1 - t / total_time)
        """
        norm = np.zeros(12, dtype=np.float64)
        norm[0] = self.state[0] / 6.0
        norm[1] = self.state[1] / 800.0
        norm[2] = self.state[2] / 0.2
        norm[3] = self.state[3] / self.M_CONC_LIMIT

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
        """Computes a stage-aware binary action mask from the state tensor.

        This mask ensures that the agent cannot apply actions that are physically
        impossible or disabled during the current operational stage.

        Args:
            state_tensor (torch.Tensor): A batch of normalized state vectors.

        Returns:
            torch.Tensor: Binary mask of shape [..., 4] where 1.0 indicates an
                active action dimension and 0.0 indicates a masked dimension.
                - dim 0 (time multiplier): Active in Inoculation/Growth.
                - dim 1 (light intensity): Active in Inoculation/Growth.
                - dim 2 (nitrate feed): Active in Inoculation/Growth.
                - dim 3 (outstream flow): Active in Harvesting only.
        """
        import torch
        # Stage one-hot at indices 4:8
        stage_0 = state_tensor[..., 4]   # Inoculation
        stage_1 = state_tensor[..., 5]   # Growth
        stage_2 = state_tensor[..., 6]   # Harvesting
        # stage_3 = state_tensor[..., 7]  # Idle — all masked

        mask_time = stage_0 + stage_1
        mask_I    = stage_0 + stage_1
        mask_Fn   = stage_0 + stage_1
        mask_Fout = stage_2

        return torch.stack([mask_time, mask_I, mask_Fn, mask_Fout], dim=-1)

    def _sigmoid_switch(self, x, center=0.0, width=2.0):
        """Calculates a smooth transition factor in the range [0, 1].

        Used to create hysteretic switching logic, primarily for blending
        Harvesting and Idle stages smoothly.

        Args:
            x (float): Input value to the sigmoid.
            center (float, optional): Center point of the sigmoid. Defaults to 0.0.
            width (float, optional): Scale width of the transition. Defaults to 2.0.

        Returns:
            float: Sigmoid output value between 0.0 and 1.0.
        """
        return 1.0 / (1.0 + np.exp(-(x - center) / max(width, 0.01)))

    def _partial_reset_reactor(self):
        """Resets the physical reactor state for a new Inoculation batch.

        This is invoked when the reactor successfully drains during Harvesting
        and there is enough episode time left to run another batch cycle.
        It preserves the episode's time, step count, cumulative rewards/penalties,
        and the nitrate supply (shared per-episode resource).
        """
        self.state = np.array([1.1, 150.0, 0.005, 0.0], dtype=np.float64)
        self.state[3] = self.state[0] * 1000.0 + self.state[1] + self.state[2]
        # NOTE: nitrate_supply is NOT reset — it is a per-episode resource
        self.remaining_frac = 1.0
        self.prev_action = np.zeros(4)


    def _physics_step(self, action):
        """Executes one control step of the physics simulation.

        This method decodes the agent's normalized action [-1, 1] into physical
        units, applies stage-based masking and constraints, deducts stage credits,
        and integrates the bioreactor state forward by `control_freq` hours.

        In the volume-less model, F_out is a fractional drain rate (h^-1) that
        only affects `remaining_frac` and harvest tracking, not concentrations.

        Args:
            action (np.ndarray): The raw action vector [time_mult, I, Fn, F_out]
                proposed by the agent, generally bounded to [-1.0, 1.0].

        Returns:
            tuple:
                - a_clipped (np.ndarray): The clipped action.
                - Fn_phys (float): The actual nitrate feed applied (mg/L/h).
                - done (bool): Whether the maximum episode time steps have been reached.
        """
        a_clipped = np.clip(action, -1.0, 1.0)
        a_scaled  = (a_clipped + 1.0) / 2.0   # [0, 1]

        # ── Action Decoding (stage-dependent) ─────────────────────────
        if self.current_stage in (0, 1):
            # Time multiplier: 0.592–1.968  (cycle ~192–480 h with 312 h credits)
            multiplier = 0.592 + a_scaled[0] * 1.376

            # Light intensity
            I_phys = self.I_MIN + a_scaled[1] * (self.I_MAX - self.I_MIN)

            # Nitrate feed (stage-dependent cap)
            fn_max = self.FN_MAX_GROWTH if self.current_stage == 0 else self.FN_MAX_PROD
            Fn_phys = a_scaled[2] * fn_max

            # Drain rate masked during growth
            F_out_rate = 0.0
        elif self.current_stage == 2:
            # Harvesting: only drain rate active
            multiplier = 1.0
            I_phys  = self.I_MIN  # baseline
            Fn_phys = 0.0
            F_out_rate = a_scaled[3] * self.FOUT_MAX  # h^-1
        else:
            # Idle: everything at baseline
            multiplier = 1.0
            I_phys  = self.I_MIN
            Fn_phys = 0.0
            F_out_rate = 0.0

        # ── Nitrate Supply Budget: Cap Fn by available supply ─────────
        if Fn_phys > 0 and self.nitrate_supply > 0:
            # Demand in concentration budget units (mg/L)
            nitrate_demand = Fn_phys * self.control_freq  # mg/L consumed
            actual_consumed = min(nitrate_demand, self.nitrate_supply)
            if actual_consumed < nitrate_demand and nitrate_demand > 0:
                Fn_phys *= (actual_consumed / nitrate_demand)
            self.nitrate_supply -= actual_consumed
        elif self.nitrate_supply <= 0:
            Fn_phys = 0.0

        # ── Hysteretic Sigmoid Blending (Harvesting → Idle) ─────────────
        blend = 0.0
        if self.current_stage == 2:
            if not self._cleanup_latch and self.remaining_frac < self.DRAIN_FRAC:
                self._cleanup_latch = True

            if self._cleanup_latch:
                # Sigmoid soft-switch: blend towards idle as drain completes
                blend = self._sigmoid_switch(
                    self.DRAIN_FRAC - self.remaining_frac, center=0.0, width=0.05)
                # Blend actions toward idle defaults
                I_phys     = I_phys * (1.0 - blend) + self.I_MIN * blend
                Fn_phys    = Fn_phys * (1.0 - blend)
                F_out_rate = F_out_rate * (1.0 - blend)

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
            self.remaining_frac = 1.0  # reset drain progress for new harvest

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
                    min_time_for_cycle = 192.0
                    time_remaining = self.total_time - self.time

                    if time_remaining >= min_time_for_cycle:
                        # Backtrack to Inoculation (stage 0) to start a new batch
                        self._partial_reset_reactor()
                        self.current_stage = 0
                        self.stage_credits = float(self.BASE_CREDITS[0])
                        self._cleanup_latch = False
                    else:
                        # Transition to Idle (stage 3)
                        self.current_stage = 3
                        self.stage_credits = 0.0


        # ── Volume-Less Integration ──────────────────────────────────
        self.state = integrate_rk4(
            self.state, I_phys, Fn_phys,
            self.dt, self.n_inner_steps)

        # ── Drain & Harvest Tracking ─────────────────────────────────
        if F_out_rate > 0:
            drain_this_step = F_out_rate * self.control_freq
            self.remaining_frac -= drain_this_step
            self.remaining_frac = max(self.remaining_frac, 0.0)
            # Harvest: concentration-equivalent of product removed (mg/L)
            self.total_cq_harvested += self.state[2] * drain_this_step

        self.time += self.control_freq
        self.time_step_count += 1

        done = self.time_step_count >= self.max_steps

        return a_clipped, Fn_phys, done