"""
Core Environment Module for the Photoproduction Bioreactor.

This module provides the base environment class `PhycocyaninEnvCore`, which
handles the mass balance, Runge-Kutta integration (envs/kinetics.py) and
stage transitions of the photobioreactor for phycocyanin production.
"""
import numpy as np
import torch

from envs.kinetics import integrate_rk4

# Stage-locked action defaults (time multiplier 1.0x, I at I_MIN, flows off).
# DEFAULT_INTENT is the pre-tanh mean of a masked dim; DEFAULT_SQUASHED is
# the squashed value sent to the environment.
DEFAULT_INTENT = (-0.4329, -10.0, -10.0, -10.0)
DEFAULT_SQUASHED = (-0.407, -1.0, -1.0, -1.0)

# Optional stress scenarios, off by default.  Each one is a failure mode the
# APN projection cannot cover by construction, so a different safety layer
# has to:
#
#   discharge    G6: an episode budget on the nitrate drained with the
#                culture.  Cumulative over every batch, so no one-step
#                projection can enforce it: the Lagrangian prices it.
#   variability  Hidden batch-to-batch kinetics (growth rate, nitrate yield,
#                synthesis rate), redrawn at every batch.  The observation
#                shows the state, not the rates, and the APN predicts with
#                the nominal ones: the GRU can infer them from the trajectory.
#   actuators    The plant runs every actuator command in every stage, as a
#                real one does; the stage locks are the controller's alone.
#                A locked actuator driven anyway is a G7 interlock breach,
#                which only the stage mask (L0) prevents.
SCENARIOS = ("discharge", "variability", "actuators")

# Actuators each stage drives (time multiplier, light, nitrate feed, outflow),
# as get_action_mask() without the nitrate-supply lock.
STAGE_ACTIVE = np.array([[True, True, True, False],     # Inoculation
                         [True, True, True, False],     # Growth
                         [False, False, False, True],   # Harvesting
                         [False, False, False, False]])  # Idle


# =============================================================================
# REINFORCEMENT LEARNING ENVIRONMENT
# =============================================================================

class PhycocyaninEnvCore:
    """Multi-stage photobioreactor RL environment core.

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

    Attributes:
        total_time (float): Maximum simulation duration in hours (1000.0).
        control_freq (float): Duration of each control action step in hours (10.0).
        max_steps (int): Maximum number of control steps per episode.
        n_stage (int): Number of operational stages (4).
        BASE_CREDITS (np.ndarray): Base time credits assigned to each stage.
        I_MIN (float): Minimum light intensity.
        I_MAX (float): Maximum light intensity.
        FN_MAX_GROWTH (float): Maximum nitrate feed rate during growth stage 0.
        FN_MAX_PROD (float): Maximum nitrate feed rate during growth stage 1.
        FOUT_MAX (float): Maximum outstream volumetric flow rate.
        V_MAX (float): Maximum reactor capacity volume.
        V_MIN (float): Minimum allowable reactor volume.
        V_INITIAL (float): Initial fill volume of the reactor.
        V_DRAIN (float): Volume threshold to trigger the cleanup/idle transition.
        V_RESET (float): Post-harvest reset volume.
        INITIAL_NITRATE_SUPPLY (float): Initial total nitrate budget.
        N_LIMIT_PATH (float): Constraint threshold for max path nitrate.
        RATIO_LIMIT (float): Constraint threshold for max cq/cx ratio.
        N_LIMIT_TERM (float): Constraint threshold for terminal nitrate.
        OVERFLOW_BUFFER_FRAC (float): Buffer fraction at which the overflow constraint activates.
        dt (float): Inner integration step duration.
        n_inner_steps (int): Number of integration steps per control step.
        time (float): Current time in the simulation.
        time_step_count (int): Current step index in the episode.
        current_stage (int): The currently active operational stage index.
        stage_credits (float): Remaining time credits for the active stage.
        state (np.ndarray): The physical state [Biomass, Nitrate, Phycocyanin, Volume].
        nitrate_supply (float): Remaining global nitrate supply.
        prev_action (np.ndarray): The action executed in the previous step.
        violation_count (int): Total number of safety violations in the episode.
        scenarios (frozenset): Active SCENARIOS.
        NITRATE_DISCHARGE_BUDGET (float): g6: nitrate drained per episode (mg), 'discharge'.
        KINETIC_SPREAD (float): Relative half-width of the hidden kinetics, 'variability'.
        INTERLOCK_TOL (float): Command on a locked actuator, in [0, 1] actuator
            units, above which it is a g7 breach ('actuators').
        kin_scale (tuple): The batch's multipliers on um, YNX and km (1.0 nominal).
        nitrate_discharged (float): Nitrate drained with the culture this episode (mg).
        interlock_breach (bool): The last step drove a stage-locked actuator ('actuators').
    """

    def __init__(self, scenarios=()):
        """Initializes the environment settings, physical boundaries, and parameters.

        Args:
            scenarios (iterable of str, optional): SCENARIOS to switch on.

        Raises:
            ValueError: On an unknown scenario.
        """
        unknown = set(scenarios) - set(SCENARIOS)
        if unknown:
            raise ValueError(f"unknown scenarios {sorted(unknown)}; choose from {SCENARIOS}")
        self.scenarios = frozenset(scenarios)

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

        # --- Scenario Config ---
        # g6: mg drained per episode.  The nominal Safe RL policy drains
        # ~9300, the Standard RL one ~6500: binding for both.
        self.NITRATE_DISCHARGE_BUDGET = 5000.0
        self.KINETIC_SPREAD = 0.20               # um, YNX, km each within ±20%
        self.INTERLOCK_TOL  = 0.025              # ILLEGAL_CMD_TOL in [0, 1] units

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
        self.g7_violation_count = 0

        # Scenario state
        self.nitrate_discharged = 0.0
        self.interlock_breach = False
        self._draw_batch_kinetics()

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
        self._draw_batch_kinetics()

    def _draw_batch_kinetics(self):
        """Draw the kinetics of a new batch: hidden under 'variability', else nominal."""
        if "variability" in self.scenarios:
            spread = self.KINETIC_SPREAD
            self.kin_scale = tuple(float(s) for s in 1.0 + np.random.uniform(-spread, spread, 3))
        else:
            self.kin_scale = (1.0, 1.0, 1.0)

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

        # ── Scenario 'actuators': no plant-side interlock ─────────────
        # Every actuator runs as commanded whatever the stage; a locked
        # physical actuator (light, feed, outflow) driven is a g7 breach.
        self.interlock_breach = False
        if "actuators" in self.scenarios:
            locked = ~STAGE_ACTIVE[self.current_stage]
            self.interlock_breach = bool(np.any(locked[1:] & (a_scaled[1:] > self.INTERLOCK_TOL)))
            I_phys = self.I_MIN + a_scaled[1] * (self.I_MAX - self.I_MIN)
            fn_max = self.FN_MAX_GROWTH if self.current_stage == 0 else self.FN_MAX_PROD
            Fn_phys = a_scaled[2] * fn_max
            F_out = a_scaled[3] * self.FOUT_MAX

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
            self.dt, self.n_inner_steps, *self.kin_scale)

        # Track harvested phycocyanin mass during cleanup outflow; outflow in
        # any other stage ('actuators') drains the culture to waste.  Either
        # way the nitrate in it is discharged (g6).
        if F_out > 0:
            drained = F_out * self.control_freq
            if self.current_stage == 2:
                self.total_cq_harvested += self.state[2] * drained
            self.nitrate_discharged += self.state[1] * drained

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