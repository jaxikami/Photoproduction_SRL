import numpy as np
from env_core import PhycocyaninEnvCore, integrate_rk4

# =============================================================================
# BENCHMARK ENVIRONMENT — Fixed-Weight Barrier + Spike Penalties
# =============================================================================

class PhycocyaninEnvBench(PhycocyaninEnvCore):
    """Benchmark photobioreactor environment with stationary constraint penalties.

    Each hard constraint is enforced via a combination of a quadratic barrier 
    penalty when approaching the limit, and a linear spike penalty when breached:
        penalty_i = W_BARRIER * buffer_depth^2          (near-boundary zone)
                  + W_gi_SPIKE * violation_magnitude     (when constraint is breached)

    No Lagrangian multipliers are used in this benchmark environment; all penalty 
    weights are fixed throughout training.

    Constraints:
        g1 (path):     Nitrate CN <= N_LIMIT_PATH (800 mg/L)        — barrier + spike
        g2 (path):     Cq/Cx ratio <= RATIO_LIMIT (0.011)           — barrier + spike
        g3 (terminal): CN <= N_LIMIT_TERM (150 mg/L) at episode end — spike
        g4 (path):     Reactor volume <= V_MAX                      — barrier + spike
        g5 (terminal): Episode MUST end in Idle stage (stage 3)     — HARD spike

    Reward structure (per step):
        reward = prod_reward
               - constraint_penalty
               - smoothing_penalty
               - raw_material_penalty
    """

    def __init__(self):
        """Initializes the benchmark environment with fixed penalty weights."""
        self.episode_count = 0
        super().__init__()

        # Disable inherited Lagrangian updates
        self.lagrange_updates_enabled = False

        # ── Fixed constraint penalty weights ───────────────────────
        # Scale: harvest_r peaks ~13/step. Spikes should dominate reward
        # at hard violations, but not be so large they prevent learning.
        self.W_BARRIER   = 5.0     # Default quadratic buffer-zone scaling coefficient
        self.W_g1_BARRIER = 40.0  # G1-specific pre-limit pressure (stronger than default barrier)
        self.W_g1_SPIKE  = 250.0  # G1: very strong linear spike to suppress nitrate-limit reward hacking
        self.W_g1_QUAD   = 900.0  # Extra superlinear term once over limit
        self.W_g2_BARRIER = 40.0  # G2-specific pre-limit pressure (matched to g1)
        self.W_g2_SPIKE  = 250.0  # G2: very strong linear spike (matched to g1)
        self.W_g2_QUAD   = 900.0  # Extra superlinear term once over limit (matched to g1)
        self.W_g3_SPIKE  = 200.0  # G3: terminal, so higher
        self.W_g4_SPIKE  = 15.0   # G4: volume overflow
        self.W_IDLE_HARD = 500.0  # G5: ~40× one harvest step — hard terminal

        # ── Reward shaping coefficients ────────────────────────────
        self.prod_coef    = 0.2    # Gentle stockpile nudge
        self.harvest_coef = 200.0  # Massive payout for physical harvesting
        self.time_penalty = 0.05   # Small operational cost
        self.smooth_coef  = 0.05   # Action-smoothing penalty coefficient
        self.raw_mat_coef = 0.1    # Nitrate feed penalty (reduced)

        # ── Buffer zone activation thresholds ─────────────────────
        # g1/g2: buffer activates at 90% of limit
        self.G1_BUFFER_START = 0.9
        self.G2_BUFFER_START = 0.95
        # g4: inherited self.OVERFLOW_BUFFER_FRAC = 0.10  (90% V_MAX)

    def reset(self, randomize=False):
        """Resets the benchmark environment for a new episode.

        Initializes states and guarantees that the initial state does not
        violate path constraints (g1, g2, g4).

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
        
        return self.get_state_norm()

    # ------------------------------------------------------------------
    # Override step() — identical physics, new reward structure
    # ------------------------------------------------------------------
    def step(self, action):
        """Executes one control step using fixed-weight barrier/spike penalties.

        Calculates the physics update, evaluates all constraint margins, applies
        fixed penalties for constraint violations, and computes the step reward.

        Args:
            action (np.ndarray): The raw action vector [time_mult, I, Fn, F_out]
                proposed by the agent, bounded to [-1.0, 1.0].

        Returns:
            tuple:
                - norm_state (np.ndarray): The new normalized state.
                - step_reward (float): The total reward accumulated over the step.
                - done (bool): Whether the episode has terminated.
                - info (dict): Diagnostic dictionary containing violation metrics.
        """
        prev_harvested = self.total_cq_harvested
        a_clipped, Fn_phys, done = self._physics_step(action)
        harvested_step = self.total_cq_harvested - prev_harvested

        # ── Production & Harvest rewards (dense) ───────────────────
        # Small reward for holding product
        prod_r = self.prod_coef * ((self.state[2] * self.state[3]) / (0.2 * self.V_MAX))
        # Large reward for draining product out of the tank
        harvest_r = harvested_step * self.harvest_coef

        # ── Fixed-weight barrier + spike constraint penalties ──────
        p_g1 = p_g2 = p_g3 = p_g4 = p_g5 = 0.0

        # Only apply instantaneous path constraints (g1, g2, g4) during Stages 0-1
        # (Growth / Production) where the agent controls I and Fn.
        # In Stage 2 (Cleanup), I is hard-coded to I_MIN and Fn=0; the agent
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
                p_g1 += self.W_g1_BARRIER * buf_depth ** 2
            if n_ratio > 1.0:
                n_viol = n_ratio - 1.0
                # Superlinear growth makes large g1 violations disproportionately expensive.
                p_g1 += self.W_g1_SPIKE * n_viol + self.W_g1_QUAD * (n_viol ** 2)
                self.violation_count    += 1
                self.g1_violation_count += 1

            # g2: Product ratio (Cq/Cx ≤ 0.011)
            ratio   = self.state[2] / (self.state[0] + 1e-8)
            q_ratio = ratio / self.RATIO_LIMIT
            if q_ratio > self.G2_BUFFER_START:
                buf_depth = min(1.0, (q_ratio - self.G2_BUFFER_START) /
                               (1.0 - self.G2_BUFFER_START))
                p_g2 += self.W_g2_BARRIER * buf_depth ** 2
            if q_ratio > 1.0:
                q_viol = q_ratio - 1.0
                # Superlinear growth makes large g2 violations disproportionately expensive.
                p_g2 += self.W_g2_SPIKE * q_viol + self.W_g2_QUAD * (q_viol ** 2)
                self.violation_count    += 1
                self.g2_violation_count += 1

            # g4: Volume overflow (V ≤ V_MAX)
            v_frac        = self.state[3] / self.V_MAX
            overflow_edge = 1.0 - self.OVERFLOW_BUFFER_FRAC
            if v_frac > overflow_edge:
                buf_depth = min(1.0, (v_frac - overflow_edge) / self.OVERFLOW_BUFFER_FRAC)
                p_g4 += self.W_BARRIER * buf_depth ** 2
            overflow_viol = max(0.0, v_frac - 1.0)
            if overflow_viol > 0:
                p_g4 += self.W_g4_SPIKE * overflow_viol
                self.violation_count    += 1
                self.g4_violation_count += 1

        constraint_penalty = p_g1 + p_g2 + p_g4

        # ── Smoothing penalty ──────────────────────────────────────
        smooth_p = self.smooth_coef * float(
            np.mean(np.square(a_clipped - self.prev_action)))
        self.prev_action = a_clipped.copy()

        # ── Raw material usage penalty ─────────────────────────────
        raw_mat_p = self.raw_mat_coef * (Fn_phys / self.FN_MAX_GROWTH)

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
        
                guiding_p = (1.0 + severity * 2.0) * action_subopt * 10.0
                p_g5 += guiding_p

        # ── Aggregate step reward ──────────────────────────────────
        step_reward = prod_r + harvest_r - constraint_penalty - smooth_p - raw_mat_p - p_g5 - self.time_penalty

        # ── Terminal checks ────────────────────────────────────────

        if done:
            # g3: Terminal nitrate (hard spike)
            t_ratio = self.state[1] / self.N_LIMIT_TERM
            if t_ratio > 1.0:
                p_g3 += self.W_g3_SPIKE * (t_ratio - 1.0)
                step_reward -= p_g3
                self.violation_count    += 1
                self.g3_violation_count += 1

            # g5: Must end in Idle stage (hard constraint — fixed large penalty)
            if self.current_stage != 3:
                current_idle_hard = self.W_IDLE_HARD
                p_g5 += current_idle_hard
                step_reward -= p_g5
                self.violation_count    += 1
                self.g5_violation_count += 1
            else:
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
        }

        return self.get_state_norm(), step_reward, done, info
