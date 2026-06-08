"""
Core Environment Module for the Photoproduction Bioreactor.

This module provides the physics simulation engine and the base environment
class `PhycocyaninEnvCore` which handles the mass balance equations,
Runge-Kutta integration, and stage transitions of the Photobioreactor
for phycocyanin production.
"""
import numpy as np
from numba import njit

# =============================================================================
# KINETIC ENGINE (Photobioreactor with Volume Tracking)
# =============================================================================

# Feed stock concentration (mg/L) — concentrated nitrate solution
C_N_STOCK = 3000.0

@njit
def calculate_rates_numba(state, I, Fn, F_out):
    """Computes instantaneous kinetic rates for the bioreactor state.

    The formulation accounts for dilution effects from feed inflow
    and product removal via outstream. This function is compiled with Numba
    for fast execution during simulation.

    Args:
        state (np.ndarray): Current physical state [Biomass Cx (g/L), 
            Nitrate CN (mg/L), Phycocyanin Cq (mg/L), Volume V (L)].
        I (float): Light intensity (umol/m^2/s).
        Fn (float): Nitrate feed concentration rate (mg/L/h).
        F_out (float): Outstream volumetric flow (L/h).

    Returns:
        np.ndarray: Array of instantaneous rates of change 
            [dCx/dt, dCN/dt, dCq/dt, dV/dt].
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
    # dCN/dt: consumption + feed term
    R[1] = -YNX * growth + F_in_vol * (C_N_STOCK - CN) / V
    # dCq/dt: synthesis – degradation – dilution from inflow
    R[2] = km * phi_Iq * Cx - (kd * Cq) / (CN + KNp) - F_in_vol * Cq / V
    # dV/dt: inflow – outflow
    R[3] = F_in_vol - F_out

    return R


@njit
def integrate_rk4(state_init, I, Fn, F_out, dt, n_steps):
    """Executes a 4th-order Runge-Kutta integration step for the bioreactor system.

    Advances the physical state of the bioreactor [Cx, CN, Cq, V] over a given
    time period using the defined instantaneous rates.

    Args:
        state_init (np.ndarray): The initial physical state before integration.
        I (float): Light intensity applied during the step.
        Fn (float): Nitrate feed rate applied during the step.
        F_out (float): Outstream flow rate during the step.
        dt (float): Inner integration time step duration (hours).
        n_steps (int): Number of integration steps to execute.

    Returns:
        np.ndarray: The updated physical state after integration.
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
        self.FOUT_MAX      = 2.0   # L/h max outstream

        # --- Reactor Volume Specs ---
        self.V_MAX        = 20.0    # L — reactor capacity
        self.V_MIN        = self.V_MAX * 0.05     # L — dry floor
        self.V_INITIAL    = 0.10 * self.V_MAX    # L — initial fill (10%)
        self.V_DRAIN      = 4.0    # L — cleanup→idle trigger
        self.V_RESET      = 0.10 * self.V_MAX   # L — post-harvest reset volume (10% of V_MAX = 2.0 L)

        # --- Global Nutrient Pool ---
        self.INITIAL_NITRATE_SUPPLY = 100000.0  # nitrate total budget

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

        # Physical state: [Cx (g/L), CN (mg/L), Cq (mg/L), V (L)]
        self.state = np.array([1.1, 150.0, 0.005, self.V_INITIAL], dtype=np.float64)

        if randomize:
            noise_factor = 0.10
            # Only randomize concentrations, not volume
            conc_noise = np.random.normal(1.0, noise_factor, size=3)
            self.state[:3] *= np.maximum(0.01, conc_noise)
            # Slight volume randomization
            self.state[3] = np.clip(
                self.state[3] * np.random.normal(1.0, 0.05),
                self.V_MIN, self.V_MAX - 5.0
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
        self.g6_violation_count = 0

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
        self.ep_g6_penalties = []

        # Phycocyanin harvested during cleanup (mass in mg)
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
                - [3] V/V_max
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
                - dim 2 (nitrate feed): Active in Inoculation/Growth, and only
                  when nitrate supply > 0.
                - dim 3 (outstream flow): Active in Harvesting only.
        """
        import torch

        # Stage one-hot at indices 4:8
        stage_0 = state_tensor[..., 4]   # Inoculation
        stage_1 = state_tensor[..., 5]   # Growth
        stage_2 = state_tensor[..., 6]   # Harvesting

        supply_available = (state_tensor[..., 10] > 0.0).float()

        mask_time = stage_0 + stage_1
        mask_I    = stage_0 + stage_1
        mask_Fn   = (stage_0 + stage_1) * supply_available
        # F_out active whenever in Harvesting stage
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
        It preserves the episode's time, step count, and cumulative rewards/penalties
        while resetting the physical concentrations, volume, and nutrient supply.

        Restriction 3: The reactor resets at 10% of V_MAX (V_RESET = 5.0 L)
        rather than the full V_INITIAL (40 L), reflecting the residual inoculum
        left in the vessel after the underflow-floor harvest.
        """
        self.state = np.array([1.1, 150.0, 0.005, self.V_RESET], dtype=np.float64)
        self.prev_action = np.zeros(4)


    def _physics_step(self, action):
        """Executes one control step of the physics simulation.

        This method decodes the agent's normalized action [-1, 1] into physical
        units, applies stage-based masking and constraints, deducts stage credits,
        and integrates the bioreactor state forward by `control_freq` hours.

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

            # Outstream masked
            F_out = 0.0
        elif self.current_stage == 2:
            # Harvesting: only outstream active
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

        # ── Hysteretic Sigmoid Blending (Harvesting → Idle) ─────────────
        blend = 0.0
        if self.current_stage == 2:
            if not self._cleanup_latch and self.state[3] < self.V_DRAIN:
                self._cleanup_latch = True

            if self._cleanup_latch:
                # We reached the drain threshold.
                # Do NOT blend F_out to 0, otherwise the agent can never empty the reactor.
                # (I_phys and Fn_phys are already at baseline during stage 2)
                pass

        # ── Time Credit Update ────────────────────────────────────────
        dt_hours = self.control_freq   # effective step duration
        if self.current_stage in (0, 1):
            self.stage_credits -= multiplier * dt_hours
        elif self.current_stage == 2:
            self.stage_credits -= dt_hours  # fixed rate in cleanup

        # ── Bioreactor Integration ─────────────────────────────────────────
        self.state = integrate_rk4(
            self.state, I_phys, Fn_phys, F_out,
            self.dt, self.n_inner_steps)

        # Track harvested phycocyanin mass during cleanup outflow
        if F_out > 0:
            self.total_cq_harvested += self.state[2] * F_out * self.control_freq

        # ── Stage Transition Logic ────────────────────────────────────

        # Restriction 2 (force-idle on nitrate exhaustion):
        # If at the END of a Inoculation or Growth step the global nitrate
        # supply is depleted, the reactor is force-transitioned to Idle
        # immediately.  This prevents any further bio-production steps from
        # running without nutrient supply.
        if self.current_stage in (0, 1):
            if self.nitrate_supply <= 0:
                # Force idle regardless of remaining credits
                self.current_stage = 3
                self.stage_credits  = 0.0
                self._cleanup_latch = False
            elif self.stage_credits <= 0:
                self.current_stage += 1
                self.stage_credits = float(self.BASE_CREDITS[self.current_stage])
                self._cleanup_latch = False

        elif self.current_stage == 2:
            # Check if volume has reached the target drain level
            harvest_complete = self.state[3] <= self.V_RESET + 0.1

            # Immediate transition bypass if harvest is complete
            if self._cleanup_latch and harvest_complete:
                self.stage_credits = 0.0

            # End of stage 2 is reached when stage_credits <= 0 AND harvest is complete
            # If credits expire but harvest is not complete, hold stage 2 (like fermentation env)
            if self.stage_credits <= 0:
                if not (self._cleanup_latch and harvest_complete):
                    self.stage_credits = 0.0
                else:
                    # End of stage 2 reached!
                    min_time_for_cycle = 192.0
                    time_remaining = self.total_time - self.time

                    if time_remaining >= min_time_for_cycle and self.nitrate_supply > 0:
                        # Backtrack to Inoculation (stage 0) to start a new batch
                        # Only restart if there is still nitrate supply (Restriction 2)
                        self._partial_reset_reactor()
                        self.current_stage = 0
                        self.stage_credits = float(self.BASE_CREDITS[0])
                        self._cleanup_latch = False
                    else:
                        # Transition to Idle (stage 3)
                        # (either time expired or nitrate supply exhausted)
                        self.current_stage = 3
                        self.stage_credits = 0.0

        self.time += self.control_freq
        self.time_step_count += 1

        done = self.time_step_count >= self.max_steps

        return a_clipped, Fn_phys, done