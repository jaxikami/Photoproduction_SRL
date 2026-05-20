import numpy as np
from env_core import PhycocyaninEnvCore, integrate_rk4

# =============================================================================
# SAFE ENVIRONMENT — Per-Constraint Lagrangian Multipliers
# =============================================================================

class PhycocyaninEnvSafe(PhycocyaninEnvCore):
    """
    Safe photobioreactor environment where each constraint is enforced via its
    own adaptive Lagrangian multiplier λ_i, updated by primal-dual gradient ascent:

        λ_i ← clip(λ_i + η * violation_i,  0, λ_max)     (on violation)
        λ_i ← λ_i * (1 - decay)                           (on satisfaction)

    The per-step Lagrangian penalty for a violated constraint is:

        penalty_i = λ_i * violation_i + BASE_SPIKE * (1 + violation_i)

    where BASE_SPIKE is a small fixed component that ensures a non-zero signal
    even before λ accumulates. A quadratic buffer zone activates near each limit
    to create smooth repulsion before the hard boundary is reached.

    Constraints
    -----------
    g1 (path):     Nitrate CN ≤ N_LIMIT_PATH (800 mg/L)         — λ_g1, buffer + spike
    g2 (path):     Cq/Cx ratio ≤ RATIO_LIMIT (0.011)            — λ_g2, buffer + spike
    g3 (terminal): CN ≤ N_LIMIT_TERM (150 mg/L) at episode end  — λ_g3, terminal spike
    g4 (path):     Reactor volume ≤ V_MAX                        — λ_g4, buffer + spike
    g6 (terminal): Episode MUST end in Idle stage (stage 3)      — λ_g6, HARD terminal spike

    g6 is enforced as a hard constraint by using a large fixed IDLE_BASE_SPIKE
    in addition to the adaptive λ_g6, making it expensive regardless of λ warmup.

    By default λ values persist across episodes to accumulate knowledge of
    constraint tightness. Set reset_lambdas_per_episode=True for evaluation.

    Reward structure (per step)
    ---------------------------
        reward = prod_reward
               - constraint_penalty          (Lagrangian + buffer)
               - smoothing_penalty
               - raw_material_penalty
    """

    def __init__(self):
        super().__init__()

        # ── Per-constraint Lagrangian multipliers ──────────────────
        self.lam_g1 = 0.0   # Path nitrate
        self.lam_g2 = 0.0   # Product ratio
        self.lam_g3 = 0.0   # Terminal nitrate
        self.lam_g4 = 0.0   # Volume overflow
        self.lam_g6 = 0.0   # Idle terminal constraint

        # ── Lagrangian update hyperparameters ──────────────────────
        self.lag_lr     = 0.5
        self.lag_decay  = 0.5 / 600.0
        self.lag_max    = 10.0
        self.lag_max_g6 = 50.0   # Higher cap for the hard idle constraint

        # ── Barrier + base spike params ────────────────────────────
        self.BASE_SPIKE           = 0.1
        self.BUFFER_COEF          = 0.02
        self.OVERFLOW_BUFFER_FRAC = 0.10   # buffer activates at 90% V_MAX
        self.IDLE_BASE_SPIKE      = 300.0  # large fixed spike for idle violation

        # ── Buffer zone activation thresholds ─────────────────────
        self.G1_BUFFER_START = 0.9
        self.G2_BUFFER_START = 0.9

        # ── Reward shaping coefficients ────────────────────────────
        self.prod_coef    = 10.0   # Dense per-step Cq concentration scale
        self.smooth_coef  = 0.005  # Action-smoothing penalty coefficient
        self.raw_mat_coef = 0.007  # Nitrate feed usage penalty coefficient

        # Whether to persist λ values across episodes (True = training default)
        self.reset_lambdas_per_episode = False

        # Disable inherited monolithic Lagrangian attributes to avoid confusion
        self.lagrange_updates_enabled = False

    # ------------------------------------------------------------------

    def reset(self, randomize=False):
        state = super().reset(randomize=randomize)
        # Ensure reset state does not violate g1, g2, g4 constraints
        self.state[1] = min(self.state[1], self.N_LIMIT_PATH * 0.9)
        self.state[3] = min(self.state[3], self.V_MAX * 0.9)
        ratio = self.state[2] / (self.state[0] + 1e-8)
        if ratio > self.RATIO_LIMIT * 0.9:
            self.state[2] = self.state[0] * self.RATIO_LIMIT * 0.9
        
        state = self.get_state_norm()

        if self.reset_lambdas_per_episode:
            self.lam_g1 = 0.0
            self.lam_g2 = 0.0
            self.lam_g3 = 0.0
            self.lam_g4 = 0.0
            self.lam_g6 = 0.0
        return state

    # ------------------------------------------------------------------

    def step(self, action):
        """
        One control step (10 h) with per-constraint Lagrangian multiplier penalties.
        """
        a_clipped, Fn_phys, done = self._physics_step(action)

        # ── Production reward (dense) ──────────────────────────────
        prod_r = self.prod_coef * self.state[2]

        # ── Per-constraint Lagrangian penalties ────────────────────
        constraint_penalty = 0.0

        # g1: Path nitrate (CN ≤ 800 mg/L)
        n_ratio = self.state[1] / self.N_LIMIT_PATH
        if n_ratio > self.G1_BUFFER_START:
            buf_depth = min(1.0, (n_ratio - self.G1_BUFFER_START) /
                           (1.0 - self.G1_BUFFER_START))
            constraint_penalty += self.BUFFER_COEF * buf_depth ** 2
        if n_ratio > 1.0:
            viol = n_ratio - 1.0
            constraint_penalty += self.lam_g1 * viol + self.BASE_SPIKE * (1.0 + viol)
            self.lam_g1 = min(self.lag_max, self.lam_g1 + self.lag_lr * viol)
            self.violation_count    += 1
            self.g1_violation_count += 1
        else:
            self.lam_g1 *= (1.0 - self.lag_decay)

        # g2: Product ratio (Cq/Cx ≤ 0.011)
        ratio   = self.state[2] / (self.state[0] + 1e-8)
        q_ratio = ratio / self.RATIO_LIMIT
        if q_ratio > self.G2_BUFFER_START:
            buf_depth = min(1.0, (q_ratio - self.G2_BUFFER_START) /
                           (1.0 - self.G2_BUFFER_START))
            constraint_penalty += self.BUFFER_COEF * buf_depth ** 2
        if q_ratio > 1.0:
            viol = q_ratio - 1.0
            constraint_penalty += self.lam_g2 * viol + self.BASE_SPIKE * (1.0 + viol)
            self.lam_g2 = min(self.lag_max, self.lam_g2 + self.lag_lr * viol)
            self.violation_count    += 1
            self.g2_violation_count += 1
        else:
            self.lam_g2 *= (1.0 - self.lag_decay)

        # g4: Volume overflow (V ≤ V_MAX)
        v_frac        = self.state[3] / self.V_MAX
        overflow_edge = 1.0 - self.OVERFLOW_BUFFER_FRAC
        if v_frac > overflow_edge:
            buf_depth = min(1.0, (v_frac - overflow_edge) / self.OVERFLOW_BUFFER_FRAC)
            constraint_penalty += self.BUFFER_COEF * buf_depth ** 2
        overflow_viol = max(0.0, v_frac - 1.0)
        if overflow_viol > 0:
            constraint_penalty += self.lam_g4 * overflow_viol + self.BASE_SPIKE * (1.0 + overflow_viol)
            self.lam_g4 = min(self.lag_max, self.lam_g4 + self.lag_lr * overflow_viol)
            self.violation_count    += 1
            self.g4_violation_count += 1
        else:
            self.lam_g4 *= (1.0 - self.lag_decay)

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
            # g3: Terminal nitrate — Lagrangian updated once at end of episode
            t_ratio = self.state[1] / self.N_LIMIT_TERM
            if t_ratio > 1.0:
                viol = t_ratio - 1.0
                # Scale terminal penalty by 100× vs path constraints
                step_reward -= self.lam_g3 * viol * 100.0 + self.BASE_SPIKE * (1.0 + viol) * 100.0
                self.lam_g3 = min(self.lag_max, self.lam_g3 + self.lag_lr * viol * 10.0)
                self.violation_count    += 1
                self.g3_violation_count += 1
            else:
                self.lam_g3 *= (1.0 - self.lag_decay * 10.0)

            # g6: Must end in Idle stage — HARD constraint
            # Enforced via large fixed base spike + adaptive λ_g6
            if self.current_stage != 3:
                step_reward -= self.lam_g6 + self.IDLE_BASE_SPIKE
                self.lam_g6 = min(self.lag_max_g6, self.lam_g6 + self.lag_lr * 20.0)
            else:
                self.lam_g6 *= (1.0 - self.lag_decay)
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
            # Expose current λ values for monitoring / logging
            "lam_g1": float(self.lam_g1),
            "lam_g2": float(self.lam_g2),
            "lam_g3": float(self.lam_g3),
            "lam_g4": float(self.lam_g4),
            "lam_g6": float(self.lam_g6),
        }

        return self.get_state_norm(), step_reward, done, info
