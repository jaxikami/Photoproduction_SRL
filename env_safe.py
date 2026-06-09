import numpy as np
from env_core import PhycocyaninEnvCore, integrate_rk4

# =============================================================================
# SAFE ENVIRONMENT — Per-Constraint Lagrangian Multipliers
# =============================================================================

class PhycocyaninEnvSafe(PhycocyaninEnvCore):
    """Safe photobioreactor environment using adaptive Lagrangian multipliers.

    Each constraint is enforced via its own adaptive Lagrangian multiplier (λ_i), 
    updated via primal-dual gradient ascent:
        λ_i <- clip(λ_i + η * violation_i,  0, λ_max)     (on violation)
        λ_i <- λ_i * (1 - decay)                          (on satisfaction)

    The per-step Lagrangian penalty for a violated constraint is:
        penalty_i = λ_i * violation_i + BASE_SPIKE * (1 + violation_i)

    Where BASE_SPIKE is a small fixed component that ensures a non-zero signal
    even before λ accumulates. A quadratic buffer zone activates near each limit
    to create smooth repulsion before the hard boundary is reached.

    Constraints:
        g1 (path):     Nitrate CN <= N_LIMIT_PATH (800 mg/L)        — λ_g1, buffer + spike
        g2 (path):     Cq/Cx ratio <= RATIO_LIMIT (0.011)           — λ_g2, buffer + spike
        g3 (terminal): CN <= N_LIMIT_TERM (150 mg/L) — λ_g3 spike applied at every
            Stage 1→2 (Growth→Harvesting) transition
        g4 (path):     Reactor volume <= V_MAX                      — λ_g4, buffer + spike
        g5 (terminal): Episode MUST end in Idle stage (stage 3)     — λ_g5, HARD terminal spike
    Note:
        g3 λ is updated every time a violation is detected at a Stage 1→2
        transition.  This is the sole enforcement point; g3 is no longer
        re-checked at episode termination.
        g5 is enforced as a hard constraint by using a large fixed IDLE_BASE_SPIKE
        in addition to the adaptive λ_g5, making it expensive regardless of λ warmup.
        By default λ values persist across episodes to accumulate knowledge of
        constraint tightness. Set `reset_lambdas_per_episode=True` for evaluation.

    Reward structure (per step):
        reward = prod_reward
               - constraint_penalty          (Lagrangian + buffer)
               - smoothing_penalty
               - raw_material_penalty
    """

    def __init__(self):
        """Initializes the safe environment and all Lagrangian hyperparameters."""
        # ── Initialise attributes used by reset() BEFORE super().__init__() ──
        # super().__init__() calls self.reset(), so these must exist first.
        self.reset_lambdas_per_episode = False
        self.episode_count = 0
        self.lag_ep_decay_g2 = 0.90    # Set early; reset() uses it
        self.lam_g1 = 0.0
        self.lam_g2 = 0.0
        self.lam_g3 = 0.0
        self.lam_g4 = 0.0
        self.lam_g5 = 0.0
        self.prev_Cx = 1.1             # Set early; step() uses it

        super().__init__()

        # ── Lagrangian update hyperparameters ──────────────────────
        # Scale: harvest_r peaks ~13/step. Penalties should dominate only
        # at severe/persistent violations, not on first contact.
        self.lag_lr     = 0.5          # Slow growth so λ ramps gradually
        self.lag_lr_g1  = 2000.0       # Scaled up for G1 (path nitrate)
        self.lag_lr_g2  = 500.0        # Gentle ramp — g2 is borderline-feasible, agent needs time to learn
        self.lag_lr_g3  = 50.0         # 100× default lag_lr — matches the 100× penalty scale-up
        self.lag_decay  = 1.0 / 600.0  # Faster decay to recover production signal after compliance
        self.lag_decay_g2 = 2.0 / 600.0  # Meaningful per-step decay (was 0.3/600)
        self.lag_ep_decay_g2 = 0.90    # Per-EPISODE multiplicative decay to prevent death spiral
        self.lag_max    = 15.0         # Default matches env_bench W_g4_SPIKE
        self.lag_max_g1 = 15000.0      # Dedicated high cap for G1 (retains G1 > G2 priority)
        self.lag_max_g2 = 500.0        # Hard cap prevents reward signal death
        self.lag_max_g3 = 20000.0       # 100× scale-up to match the 100× G3 penalty multiplier
        self.lag_max_g5 = 100.0        # G5 cap (hard constraint)

        # ── Barrier + base spike params ────────────────────────────
        # BASE_SPIKE fires even at λ=0; must be painful but not catastrophic.
        self.BASE_SPIKE           = 3.0   # Reduced from 2.0 to match Fermentation dynamics closer (scaled up)
        self.BUFFER_COEF          = 50.0  # Matches env_bench W_BARRIER (scaled up)
        self.OVERFLOW_BUFFER_FRAC = 0.10  # buffer activates at 90% V_MAX
        self.IDLE_BASE_SPIKE      = 5000.0 # Hard idle spike: ~40× one harvest step (scaled up)

        # ── Buffer zone activation thresholds ─────────────────────
        self.G1_BUFFER_START = 0.95
        self.G2_BUFFER_START = 0.95  # Barrier penalty starts at 95% of the ratio limit

        # ── Reward shaping coefficients ────────────────────────────
        self.prod_coef    = 0.4    # Gentle stockpile nudge
        self.harvest_coef = 400.0  # Massive payout for physical harvesting
        self.time_penalty = 0.05   # Small operational cost (was 0.2, too aggressive)
        self.smooth_coef  = 0.05   # Action-smoothing penalty coefficient
        self.raw_mat_coef = 0.1    # Nitrate feed penalty (was 0.5; conflicted with g2 avoidance)

        # Disable inherited monolithic Lagrangian attributes to avoid confusion
        self.lagrange_updates_enabled = False

    # ------------------------------------------------------------------

    def reset(self, randomize=False):
        """Resets the safe environment for a new episode.

        Optionally resets the Lagrangian multipliers if `reset_lambdas_per_episode` 
        is True. Ensures the starting state does not violate path constraints.

        Args:
            randomize (bool, optional): Whether to inject initial state noise.
                Defaults to False.

        Returns:
            np.ndarray: The normalized initial observation vector.
        """
        self.episode_count += 1
        state = super().reset(randomize=randomize)
        # Ensure reset state does not violate g1, g2, g4 constraints
        self.state[1] = min(self.state[1], self.N_LIMIT_PATH * 0.9)
        self.state[3] = min(self.state[3], self.V_MAX * 0.9)
        ratio = self.state[2] / (self.state[0] + 1e-8)
        if ratio > self.RATIO_LIMIT * 0.7:
            self.state[2] = self.state[0] * self.RATIO_LIMIT * 0.7

        # Track previous Cx for growth bonus (g2 avoidance shaping)
        self.prev_Cx = self.state[0]
        
        state = self.get_state_norm()

        if self.reset_lambdas_per_episode:
            self.lam_g1 = 0.0
            self.lam_g2 = 0.0
            self.lam_g3 = 0.0
            self.lam_g4 = 0.0
            self.lam_g5 = 0.0
        else:
            # Per-episode decay to prevent death spiral from early exploration
            self.lam_g2 *= self.lag_ep_decay_g2
        self.g6_violation_count = 0
        return state

    # ------------------------------------------------------------------

    def step(self, action):
        """Executes one control step using adaptive Lagrangian penalties.

        Evaluates the current state margins and updates the corresponding
        Lagrangian multipliers dynamically based on violations.

        g3 is evaluated on every step where a Stage 1→2 (Growth→Harvesting)
        transition occurs, i.e. whenever the Growth stage ends and Harvesting
        begins.  This is the sole enforcement point for the terminal nitrate
        constraint; it is no longer re-checked at episode termination.
        λ_g3 is updated on each detected violation, allowing it to accumulate
        pressure across multiple batches within the same episode.

        Args:
            action (np.ndarray): The raw action vector [time_mult, I, Fn, F_out]
                proposed by the agent, bounded to [-1.0, 1.0].

        Returns:
            tuple:
                - norm_state (np.ndarray): The new normalized state.
                - step_reward (float): The total reward accumulated over the step.
                - done (bool): Whether the episode has terminated.
                - info (dict): Diagnostic dictionary containing violation metrics
                  and current multiplier values.
        """
        prev_harvested = self.total_cq_harvested
        prev_stage = self.current_stage          # capture BEFORE physics step
        a_clipped, Fn_phys, done = self._physics_step(action)
        harvested_step = self.total_cq_harvested - prev_harvested
        stage_transitioned_to_cleanup = (prev_stage == 1 and self.current_stage == 2)

        # ── Production & Harvest rewards (dense) ───────────────────
        # Small reward for holding product
        prod_r = self.prod_coef * ((self.state[2] * self.state[3]) / (0.2 * self.V_MAX))
        # Large reward for draining product out of the tank
        harvest_r = harvested_step * self.harvest_coef

        # ── Per-constraint Lagrangian penalties ────────────────────
        p_g1 = p_g2 = p_g3 = p_g4 = p_g5 = 0.0
        force_idle_signal = 0.0

        # Only apply instantaneous path constraints (g1, g2, g4) during Stages 0-1
        # (Inoculation / Growth) where the agent controls I and Fn.
        # In Stage 2 (Harvesting), I is hard-coded to I_MIN and Fn=0; the agent
        # only controls F_out, which doesn't affect concentrations.  Cq/Cx
        # inevitably drifts up as product synthesis continues while growth
        # stalls from CN depletion — penalising this is unfair.
        # In Stage 3 (Idle), all controls are masked.
        if self.current_stage in (0, 1):
            # g1: Path nitrate (CN ≤ 800 mg/L)
            n_ratio = self.state[1] / self.N_LIMIT_PATH
            if n_ratio > self.G1_BUFFER_START:
                buf_depth = min(1.0, (n_ratio - self.G1_BUFFER_START) /
                               (1.0 - self.G1_BUFFER_START))
                p_g1 += self.BUFFER_COEF * buf_depth ** 2
            if n_ratio > 1.0:
                viol = n_ratio - 1.0
                p_g1 += self.lam_g1 * viol + self.BASE_SPIKE * (1.0 + viol)
                self.lam_g1 = min(self.lag_max_g1, self.lam_g1 + self.lag_lr_g1 * viol)
                self.violation_count    += 1
                self.g1_violation_count += 1
            else:
                self.lam_g1 *= (1.0 - self.lag_decay)

            # g2: Product ratio (Cq/Cx ≤ 0.011) — PROXIMITY-BASED SHAPING
            ratio   = self.state[2] / (self.state[0] + 1e-8)
            q_ratio = ratio / self.RATIO_LIMIT

            if q_ratio > self.G2_BUFFER_START:
                # Normalized distance into danger zone: 0 at buffer start → 1 at limit
                buf_depth = min(1.0, (q_ratio - self.G2_BUFFER_START) /
                               (1.0 - self.G2_BUFFER_START))
                # Quadratic ramp — penalty grows quadratically as ratio approaches limit
                p_g2 += self.BUFFER_COEF * buf_depth ** 2

            if q_ratio > 1.0:
                # Beyond limit: linear penalty proportional to violation magnitude
                viol = q_ratio - 1.0
                self.lam_g2 = min(self.lag_max_g2, self.lam_g2 + self.lag_lr_g2 * viol)
                p_g2 += self.lam_g2 * viol + self.BASE_SPIKE * (1.0 + viol)
                # Cap per-step g2 penalty to preserve learning signal
                p_g2 = min(p_g2, 10.0)
                self.violation_count    += 1
                self.g2_violation_count += 1
            else:
                self.lam_g2 *= (1.0 - self.lag_decay_g2)

            # g4: Volume overflow (V ≤ V_MAX)
            v_frac        = self.state[3] / self.V_MAX
            overflow_edge = 1.0 - self.OVERFLOW_BUFFER_FRAC
            if v_frac > overflow_edge:
                buf_depth = min(1.0, (v_frac - overflow_edge) / self.OVERFLOW_BUFFER_FRAC)
                p_g4 += self.BUFFER_COEF * buf_depth ** 2
            overflow_viol = max(0.0, v_frac - 1.0)
            if overflow_viol > 0:
                p_g4 += self.lam_g4 * overflow_viol + self.BASE_SPIKE * (1.0 + overflow_viol)
                self.lam_g4 = min(self.lag_max, self.lam_g4 + self.lag_lr * overflow_viol)
                self.violation_count    += 1
                self.g4_violation_count += 1
            else:
                self.lam_g4 *= (1.0 - self.lag_decay)

        # Restriction 2: If nitrate supply is exhausted, force idle via penalty
        if self.nitrate_supply <= 0 and self.current_stage != 3:
            force_idle_signal = 100.0

        constraint_penalty = p_g1 + p_g2 + p_g4

        # ── Smoothing penalty ──────────────────────────────────────
        smooth_p = self.smooth_coef * float(
            np.mean(np.square(a_clipped - self.prev_action)))
        self.prev_action = a_clipped.copy()

        # ── Raw material usage penalty ─────────────────────────────
        raw_mat_p = self.raw_mat_coef * (Fn_phys / self.FN_MAX_GROWTH)

        self.prev_Cx = self.state[0]

        # ── G5 Directional Guiding ─────────────────────────────────────
        time_remaining = self.total_time - self.time
        if self.current_stage != 3:
            min_time_to_idle = 0.0
            if self.current_stage == 0:
                min_time_to_idle = (self.stage_credits / 1.968) + (self.BASE_CREDITS[1] / 1.968) + max(0.0, (self.state[3] - (self.V_DRAIN - 2.0)) / self.FOUT_MAX)
            elif self.current_stage == 1:
                min_time_to_idle = (self.stage_credits / 1.968) + max(0.0, (self.state[3] - (self.V_DRAIN - 2.0)) / self.FOUT_MAX)
            elif self.current_stage == 2:
                min_time_to_idle = max(0.0, (self.state[3] - (self.V_DRAIN - 2.0)) / self.FOUT_MAX)
        
            g5_buffer = 30.0  # Danger zone width (hours)
            margin = time_remaining - min_time_to_idle
            if margin < g5_buffer:
                severity = max(0.0, (g5_buffer - margin) / g5_buffer)
                
                a_scaled = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
                action_subopt = 0.0
                if self.current_stage in (0, 1):
                    action_subopt = 1.0 - a_scaled[0]
                elif self.current_stage == 2:
                    action_subopt = 1.0 - a_scaled[3]
        
                guiding_p = (1.0 + severity * 2.0) * action_subopt * 30.0
                p_g5 += guiding_p

        # ── Aggregate step reward ──────────────────────────────────
        step_reward = prod_r + harvest_r - constraint_penalty - smooth_p - raw_mat_p - p_g5 - force_idle_signal - self.time_penalty

        # ── Terminal checks ────────────────────────────────────────

        # g3: Nitrate check at Stage 1→2 (Growth→Harvesting) transition only
        # λ_g3 is updated on each detected violation.
        if stage_transitioned_to_cleanup:
            t_ratio = self.state[1] / self.N_LIMIT_TERM
            if t_ratio > 1.0:
                viol = t_ratio - 1.0
                # Use lag_max_g3 specifically
                p_g3 += 100.0 * (self.lam_g3 * viol + self.BASE_SPIKE * (1.0 + viol))
                step_reward -= p_g3
                self.lam_g3 = min(self.lag_max_g3, self.lam_g3 + self.lag_lr_g3 * viol)
                self.violation_count    += 1
                self.g3_violation_count += 1
            else:
                self.lam_g3 *= (1.0 - self.lag_decay)

        if done:

            # g5: Must end in Idle stage — HARD constraint
            # Enforced via large fixed base spike + adaptive λ_g5
            if self.current_stage != 3:
                current_idle_spike = self.IDLE_BASE_SPIKE
                extra_p_g5 = self.lam_g5 + current_idle_spike
                p_g5 += extra_p_g5
                step_reward -= extra_p_g5
                self.lam_g5 = min(self.lag_max_g5, self.lam_g5 + self.lag_lr * 5.0)
                self.violation_count    += 1
                self.g5_violation_count += 1
            else:
                self.lam_g5 *= (1.0 - self.lag_decay)
                step_reward += 50.0  # Idle completion bonus

        # ── Metrics bookkeeping ────────────────────────────────────
        self.ep_total_reward += step_reward
        self.ep_rewards.append(step_reward)
        self.ep_prod_rewards.append(prod_r)
        self.ep_harvest_rewards.append(harvest_r)
        self.ep_smooth_penalties.append(smooth_p)
        self.ep_constraint_penalties.append(constraint_penalty)
        self.ep_raw_mat_penalties.append(raw_mat_p)
        self.ep_g1_penalties.append(p_g1)
        self.ep_g2_penalties.append(p_g2)
        self.ep_g3_penalties.append(p_g3)
        self.ep_g4_penalties.append(p_g4)
        self.ep_g5_penalties.append(p_g5)
        self.ep_g6_penalties.append(0.0)

        info = {
            "avg_reward":             float(np.mean(self.ep_rewards)),
            "total_reward":           self.ep_total_reward,
            "avg_prod_reward":        float(np.mean(self.ep_prod_rewards)),
            "avg_harvest_reward":     float(np.mean(self.ep_harvest_rewards)),
            "avg_smooth_penalty":     float(np.mean(self.ep_smooth_penalties)),
            "avg_constraint_penalty": float(np.mean(self.ep_constraint_penalties)),
            "avg_raw_mat_penalty":    float(np.mean(self.ep_raw_mat_penalties)),
            "avg_g1_penalty":         float(np.mean(self.ep_g1_penalties)),
            "avg_g2_penalty":         float(np.mean(self.ep_g2_penalties)),
            "avg_g3_penalty":         float(np.mean(self.ep_g3_penalties)),
            "avg_g4_penalty":         float(np.mean(self.ep_g4_penalties)),
            "avg_g5_penalty":         float(np.mean(self.ep_g5_penalties)),
            "violation_count":        self.violation_count,
            "g1_violation_count":     self.g1_violation_count,
            "g2_violation_count":     self.g2_violation_count,
            "g3_violation_count":     self.g3_violation_count,
            "g4_violation_count":     self.g4_violation_count,
            "g5_violation_count":     self.g5_violation_count,
            "current_stage":          self.current_stage,
            "volume":                 float(self.state[3]),
            "nitrate_supply":         float(self.nitrate_supply),
            "total_cq_harvested":     float(self.total_cq_harvested),
            # Expose current λ values for monitoring / logging
            "lam_g1": float(self.lam_g1),
            "lam_g2": float(self.lam_g2),
            "lam_g3": float(self.lam_g3),
            "lam_g4": float(self.lam_g4),
            "lam_g5": float(self.lam_g5),
        }

        return self.get_state_norm(), step_reward, done, info
