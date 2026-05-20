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
        super().__init__()

        # Disable inherited Lagrangian updates
        self.lagrange_updates_enabled = False

        # ── Fixed constraint penalty weights ───────────────────────
        self.W_BARRIER   = 0.05    # Quadratic buffer-zone scaling coefficient
        self.W_g1_SPIKE  = 5.0    # Path nitrate hard spike
        self.W_g2_SPIKE  = 5.0    # Product ratio hard spike
        self.W_g3_SPIKE  = 200.0  # Terminal nitrate hard spike
        self.W_g4_SPIKE  = 5.0    # Volume overflow hard spike
        self.W_IDLE_HARD = 500.0  # Must end in Idle — hard terminal penalty

        # ── Reward shaping coefficients ────────────────────────────
        self.prod_coef    = 10.0   # Dense per-step Cq concentration scale
        self.smooth_coef  = 0.005  # Action-smoothing penalty coefficient
        self.raw_mat_coef = 0.007  # Nitrate feed usage penalty coefficient

        # ── Buffer zone activation thresholds ─────────────────────
        # g1/g2: buffer activates at 90% of limit
        self.G1_BUFFER_START = 0.9
        self.G2_BUFFER_START = 0.9
        # g4: inherited self.OVERFLOW_BUFFER_FRAC = 0.10  (90% V_MAX)

    def reset(self, randomize=False):
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
        prod_r = self.prod_coef * self.state[2]

        # ── Fixed-weight barrier + spike constraint penalties ──────
        constraint_penalty = 0.0

        # g1: Path nitrate (CN ≤ 800 mg/L)
        n_ratio = self.state[1] / self.N_LIMIT_PATH
        if n_ratio > self.G1_BUFFER_START:
            buf_depth = min(1.0, (n_ratio - self.G1_BUFFER_START) /
                           (1.0 - self.G1_BUFFER_START))
            constraint_penalty += self.W_BARRIER * buf_depth ** 2
        if n_ratio > 1.0:
            constraint_penalty += self.W_g1_SPIKE * (n_ratio - 1.0)
            self.violation_count    += 1
            self.g1_violation_count += 1

        # g2: Product ratio (Cq/Cx ≤ 0.011)
        ratio   = self.state[2] / (self.state[0] + 1e-8)
        q_ratio = ratio / self.RATIO_LIMIT
        if q_ratio > self.G2_BUFFER_START:
            buf_depth = min(1.0, (q_ratio - self.G2_BUFFER_START) /
                           (1.0 - self.G2_BUFFER_START))
            constraint_penalty += self.W_BARRIER * buf_depth ** 2
        if q_ratio > 1.0:
            constraint_penalty += self.W_g2_SPIKE * (q_ratio - 1.0)
            self.violation_count    += 1
            self.g2_violation_count += 1

        # g4: Volume overflow (V ≤ V_MAX)
        v_frac        = self.state[3] / self.V_MAX
        overflow_edge = 1.0 - self.OVERFLOW_BUFFER_FRAC
        if v_frac > overflow_edge:
            buf_depth = min(1.0, (v_frac - overflow_edge) / self.OVERFLOW_BUFFER_FRAC)
            constraint_penalty += self.W_BARRIER * buf_depth ** 2
        overflow_viol = max(0.0, v_frac - 1.0)
        if overflow_viol > 0:
            constraint_penalty += self.W_g4_SPIKE * overflow_viol
            self.violation_count    += 1
            self.g4_violation_count += 1

        # ── Smoothing penalty ──────────────────────────────────────
        smooth_p = self.smooth_coef * float(
            np.mean(np.square(a_clipped - self.prev_action)))
        self.prev_action = a_clipped.copy()

        # ── Raw material usage penalty ─────────────────────────────
        raw_mat_p = self.raw_mat_coef * Fn_phys

        # ── Aggregate step reward ──────────────────────────────────
        step_reward = prod_r - constraint_penalty - smooth_p - raw_mat_p

        # ── Terminal checks ────────────────────────────────────────

        if done:
            # g3: Terminal nitrate (hard spike)
            t_ratio = self.state[1] / self.N_LIMIT_TERM
            if t_ratio > 1.0:
                step_reward -= self.W_g3_SPIKE * (t_ratio - 1.0)
                self.violation_count    += 1
                self.g3_violation_count += 1

            # g6: Must end in Idle stage (hard constraint — fixed large penalty)
            if self.current_stage != 3:
                step_reward -= self.W_IDLE_HARD
            else:
                step_reward += 50.0  # Idle completion bonus

        # ── Metrics bookkeeping ────────────────────────────────────
        self.ep_total_reward += step_reward
        self.ep_rewards.append(step_reward)
        self.ep_prod_rewards.append(prod_r)
        self.ep_smooth_penalties.append(smooth_p)
        self.ep_constraint_penalties.append(constraint_penalty)

        info = {
            "avg_reward":             float(np.mean(self.ep_rewards)),
            "total_reward":           self.ep_total_reward,
            "avg_prod_reward":        float(np.mean(self.ep_prod_rewards)),
            "avg_smooth_penalty":     float(np.mean(self.ep_smooth_penalties)),
            "avg_constraint_penalty": float(np.mean(self.ep_constraint_penalties)),
            "violation_count":        self.violation_count,
            "g1_violation_count":     self.g1_violation_count,
            "g2_violation_count":     self.g2_violation_count,
            "g3_violation_count":     self.g3_violation_count,
            "g4_violation_count":     self.g4_violation_count,
            "current_stage":          self.current_stage,
            "volume":                 float(self.state[3]),
            "nitrate_supply":         float(self.nitrate_supply),
            "total_cq_harvested":     float(self.total_cq_harvested),
        }

        return self.get_state_norm(), step_reward, done, info
