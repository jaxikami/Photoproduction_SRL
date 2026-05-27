import numpy as np
from env_core import PhycocyaninEnvCore, integrate_rk4

# =============================================================================
# BENCHMARK ENVIRONMENT — Fixed-Weight Barrier + Spike Penalties
# =============================================================================

class PhycocyaninEnvBench(PhycocyaninEnvCore):
    """
    Benchmark photobioreactor environment with stationary (non-adaptive)
    constraint penalties. Each hard constraint is enforced via:

        penalty_i = W_BARRIER * buffer_depth^2          (near-boundary zone)
                  + W_gi_SPIKE * violation_magnitude     (when constraint is breached)

    No Lagrangian multipliers — all penalty weights are fixed throughout training.

    Constraints
    -----------
    g1 (path):     Nitrate CN ≤ N_LIMIT_PATH (800 mg/L)         — barrier + spike
    g2 (path):     Cq/Cx ratio ≤ RATIO_LIMIT (0.011)            — barrier + spike
    g3 (terminal): CN ≤ N_LIMIT_TERM (150 mg/L) at episode end  — spike
    g4 (path):     Reactor volume ≤ V_MAX                        — barrier + spike
    g6 (terminal): Episode MUST end in Idle stage (stage 3)      — HARD spike

    Reward structure (per step)
    ---------------------------
        reward = prod_reward
               - constraint_penalty
               - smoothing_penalty
               - raw_material_penalty
    """

    def __init__(self):
        self.episode_count = 0
        super().__init__()

        # Disable inherited Lagrangian updates
        self.lagrange_updates_enabled = False

        # ── Fixed constraint penalty weights ───────────────────────
        self.W_BARRIER   = 20.0    # Quadratic buffer-zone scaling coefficient  (~1 at full depth)
        self.W_g1_SPIKE  = 200.0  # Path nitrate hard spike                    (~10 per unit violation)
        self.W_g2_SPIKE  = 200.0  # Product ratio hard spike                   (~10 per unit violation)
        self.W_g3_SPIKE  = 1000.0 # Terminal nitrate hard spike
        self.W_g4_SPIKE  = 200.0  # Volume overflow hard spike                 (~10 per unit violation)
        self.W_IDLE_HARD = 1000.0 # Must end in Idle — hard terminal penalty

        # ── Reward shaping coefficients ────────────────────────────
        self.prod_coef    = 4.0    # Dense per-step Cq scale  (4 * (Cq/0.2) ≈ 2 at Cq=0.1)
        self.smooth_coef  = 0.5   # Action-smoothing penalty coefficient
        self.raw_mat_coef = 3.0    # Nitrate feed penalty  (0.3 * Fn/FN_MAX ≈ 0.15 at half-max)

        # ── Buffer zone activation thresholds ─────────────────────
        # g1/g2: buffer activates at 90% of limit
        self.G1_BUFFER_START = 0.9
        self.G2_BUFFER_START = 0.9
        # g4: inherited self.OVERFLOW_BUFFER_FRAC = 0.10  (90% V_MAX)

    def reset(self, randomize=False):
        self.episode_count += 1
        state = super().reset(randomize=randomize)
        # Ensure reset state does not violate g1, g2, g4 constraints
        self.state[1] = min(self.state[1], self.N_LIMIT_PATH * 0.9)
        self.state[3] = min(self.state[3], self.V_MAX * 0.9)
        ratio = self.state[2] / (self.state[0] + 1e-8)
        if ratio > self.RATIO_LIMIT * 0.9:
            self.state[2] = self.state[0] * self.RATIO_LIMIT * 0.9
        
        return self.get_state_norm()

    # ------------------------------------------------------------------
    # Override step() — identical physics, new reward structure
    # ------------------------------------------------------------------
    def step(self, action):
        """
        One control step (10 h) with fixed-weight barrier + spike penalties.
        """
        a_clipped, Fn_phys, done = self._physics_step(action)

        # ── Production reward (dense) ──────────────────────────────
        # Reward based on total mass (Cq * Volume) relative to max possible mass
        prod_r = self.prod_coef * ((self.state[2] * self.state[3]) / (0.2 * self.V_MAX))

        # ── Fixed-weight barrier + spike constraint penalties ──────
        p_g1 = p_g2 = p_g3 = p_g4 = p_g5 = p_g6 = 0.0

        # g1: Path nitrate (CN ≤ 800 mg/L)
        n_ratio = self.state[1] / self.N_LIMIT_PATH
        if n_ratio > self.G1_BUFFER_START:
            buf_depth = min(1.0, (n_ratio - self.G1_BUFFER_START) /
                           (1.0 - self.G1_BUFFER_START))
            p_g1 += self.W_BARRIER * buf_depth ** 2
        if n_ratio > 1.0:
            p_g1 += self.W_g1_SPIKE * (n_ratio - 1.0)
            self.violation_count    += 1
            self.g1_violation_count += 1

        # g2: Product ratio (Cq/Cx ≤ 0.011)
        ratio   = self.state[2] / (self.state[0] + 1e-8)
        q_ratio = ratio / self.RATIO_LIMIT
        if q_ratio > self.G2_BUFFER_START:
            buf_depth = min(1.0, (q_ratio - self.G2_BUFFER_START) /
                           (1.0 - self.G2_BUFFER_START))
            p_g2 += self.W_BARRIER * buf_depth ** 2
        if q_ratio > 1.0:
            p_g2 += self.W_g2_SPIKE * (q_ratio - 1.0)
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

        constraint_penalty = p_g1 + p_g2 + p_g4 + p_g5

        # ── Smoothing penalty ──────────────────────────────────────
        smooth_p = self.smooth_coef * float(
            np.mean(np.square(a_clipped - self.prev_action)))
        self.prev_action = a_clipped.copy()

        # ── Raw material usage penalty ─────────────────────────────
        raw_mat_p = self.raw_mat_coef * (Fn_phys / self.FN_MAX_GROWTH)

        # ── G6 Directional Guiding ─────────────────────────────────────
        time_remaining = self.total_time - self.time
        if self.current_stage != 3:
            min_time_to_idle = 0.0
            if self.current_stage == 0:
                min_time_to_idle = (self.stage_credits / 2.0) + (self.BASE_CREDITS[1] / 2.0) + max(0.0, (self.state[3] - (self.V_DRAIN - 2.0)) / self.FOUT_MAX)
            elif self.current_stage == 1:
                min_time_to_idle = (self.stage_credits / 2.0) + max(0.0, (self.state[3] - (self.V_DRAIN - 2.0)) / self.FOUT_MAX)
            elif self.current_stage == 2:
                min_time_to_idle = max(0.0, (self.state[3] - (self.V_DRAIN - 2.0)) / self.FOUT_MAX)
        
            g6_buffer = 30.0  # Danger zone width (hours)
            margin = time_remaining - min_time_to_idle
            if margin < g6_buffer:
                severity = max(0.0, (g6_buffer - margin) / g6_buffer)
                
                a_scaled = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
                action_subopt = 0.0
                if self.current_stage in (0, 1):
                    action_subopt = 1.0 - a_scaled[0]
                elif self.current_stage == 2:
                    action_subopt = 1.0 - a_scaled[3]
        
                guiding_p = (1.0 + severity * 2.0) * action_subopt * 10.0
                p_g6 += guiding_p

        # ── Aggregate step reward ──────────────────────────────────
        step_reward = prod_r - constraint_penalty - smooth_p - raw_mat_p - p_g6

        # ── Terminal checks ────────────────────────────────────────

        if done:
            # g3: Terminal nitrate (hard spike)
            t_ratio = self.state[1] / self.N_LIMIT_TERM
            if t_ratio > 1.0:
                p_g3 += self.W_g3_SPIKE * (t_ratio - 1.0)
                step_reward -= p_g3
                self.violation_count    += 1
                self.g3_violation_count += 1

            # g6: Must end in Idle stage (hard constraint — fixed large penalty)
            if self.current_stage != 3:
                # Anneal the idle penalty from 0 to W_IDLE_HARD over 25000 episodes
                current_idle_hard = min(self.W_IDLE_HARD, self.W_IDLE_HARD * (self.episode_count / 25000.0))
                p_g6 += current_idle_hard
                step_reward -= p_g6
            else:
                step_reward += 50.0  # Idle completion bonus

            # Terminal harvest bonus: reward sum of phycocyanin produced across cycles
            step_reward += self.total_cq_harvested * 50.0  # Scale total harvest bonus

        # ── Metrics bookkeeping ────────────────────────────────────
        self.ep_total_reward += step_reward
        self.ep_rewards.append(step_reward)
        self.ep_prod_rewards.append(prod_r)
        self.ep_smooth_penalties.append(smooth_p)
        self.ep_constraint_penalties.append(constraint_penalty)
        self.ep_raw_mat_penalties.append(raw_mat_p)
        self.ep_g1_penalties.append(p_g1)
        self.ep_g2_penalties.append(p_g2)
        self.ep_g3_penalties.append(p_g3)
        self.ep_g4_penalties.append(p_g4)
        self.ep_g5_penalties.append(p_g5)
        self.ep_g6_penalties.append(p_g6)

        info = {
            "avg_reward":             float(np.mean(self.ep_rewards)),
            "total_reward":           self.ep_total_reward,
            "avg_prod_reward":        float(np.mean(self.ep_prod_rewards)),
            "avg_smooth_penalty":     float(np.mean(self.ep_smooth_penalties)),
            "avg_constraint_penalty": float(np.mean(self.ep_constraint_penalties)),
            "avg_raw_mat_penalty":    float(np.mean(self.ep_raw_mat_penalties)),
            "avg_g1_penalty":         float(np.mean(self.ep_g1_penalties)),
            "avg_g2_penalty":         float(np.mean(self.ep_g2_penalties)),
            "avg_g3_penalty":         float(np.mean(self.ep_g3_penalties)),
            "avg_g4_penalty":         float(np.mean(self.ep_g4_penalties)),
            "avg_g5_penalty":         float(np.mean(self.ep_g5_penalties)),
            "avg_g6_penalty":         float(np.mean(self.ep_g6_penalties)),
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
