"""Per-episode metrics shared by training and evaluation runs."""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from envs.plant import PhycocyaninEnvCore
from experiments.ablation.config import ILLEGAL_CMD_TOL, SEVERITY_TYPES, VIOLATION_NAMES

TRAIN_LOG_FIELDS = ("episode", "return", "harvested", "violation_steps", "violation_ids",
                    "ended_idle", "lambda_max", "apn_noop_frac", "apn_avg_iters",
                    "illegal_cmd_rate", "seconds")


class EpisodeStats:
    """Accumulates the per-episode metrics shared by training and evaluation.

    Severity is measured on the plant's true state with the same rules the
    environment uses to count a violation: G1, G2 and G4 after a step that ends
    in Inoculation or Growth, G3 on the Growth -> Harvesting transition.

    Attributes:
        events: Violation events of each VIOLATION_NAMES constraint.
        depth_sum: Summed relative depth of each SEVERITY_TYPES constraint's events.
        peak: Worst relative depth of each SEVERITY_TYPES constraint's events.
    """

    def __init__(self):
        """Start an empty episode."""
        self.ret = 0.0
        self.depth_sum = dict.fromkeys(SEVERITY_TYPES, 0.0)
        self.peak = dict.fromkeys(SEVERITY_TYPES, 0.0)
        self.events = dict.fromkeys(VIOLATION_NAMES.values(), 0)
        self.last_info = {}
        self.prev_stage = 0
        self.locked = 0
        self.illegal = 0

    def _depth(self, name, depth):
        """Record one event's relative exceedance (ignored unless positive)."""
        if depth > 0.0:
            self.depth_sum[name] += depth
            self.peak[name] = max(self.peak[name], depth)

    def record(self, env, obs, raw_intent, action, reward, info):
        """Add one control step.

        Args:
            env: Environment after the step (its true physical state).
            obs: Observation the action was selected on, [state_dim].
            raw_intent: Raw pre-tanh intent returned by select_action().
            action: Action sent to the plant (after projection and interlock).
            reward: Step reward.
            info: Info dict from env.step().
        """
        # select_action() overwrites stage-locked dims with their defaults, so a
        # gap there between the raw command and the action sent is an illegal
        # command the interlock blocked (always zero for a masked policy).
        locked = PhycocyaninEnvCore.get_action_mask(
            torch.as_tensor(obs, dtype=torch.float32)).numpy() == 0
        self.locked += int(locked.sum())
        self.illegal += int((locked & (np.abs(np.tanh(raw_intent) - action) > ILLEGAL_CMD_TOL)).sum())

        cx, cn, cq, v = env.state
        stage = info["current_stage"]
        if stage in (0, 1):
            self._depth("path_nitrate", cn / env.N_LIMIT_PATH - 1.0)
            self._depth("product_ratio", (cq / (cx + 1e-8)) / env.RATIO_LIMIT - 1.0)
            self._depth("overflow", v / env.V_MAX - 1.0)
        if self.prev_stage == 1 and stage == 2:
            self._depth("terminal_nitrate", cn / env.N_LIMIT_TERM - 1.0)
        self.prev_stage = stage

        self.ret += float(reward)
        self.last_info = info

    def finish(self, env, agent):
        """Close the episode and return its metrics.

        Args:
            env: Environment at the end of the episode.
            agent: Agent whose projection counters cover this episode.

        Returns:
            Dict keyed by TRAIN_LOG_FIELDS except 'episode' and 'seconds'.
        """
        proj = agent.get_and_reset_proj_stats()
        info = self.last_info
        for vid, name in VIOLATION_NAMES.items():
            self.events[name] = int(info.get(f"g{vid}_violation_count", 0))
        lambda_max = max(info.get(f"lam_g{k}", 0.0) for k in range(1, 6))
        return {
            "return": self.ret,
            "harvested": float(info.get("total_cq_harvested", 0.0)),
            "violation_steps": int(info.get("violation_count", 0)),
            "violation_ids": "".join(str(v) for v, n in VIOLATION_NAMES.items()
                                     if self.events[n] > 0) or "0",
            "ended_idle": int(env.current_stage == 3),
            "lambda_max": lambda_max,
            "apn_noop_frac": proj["noop"] / proj["calls"] if proj["calls"] else None,
            "apn_avg_iters": proj["avg_it"] if proj["calls"] > proj["noop"] else None,
            "illegal_cmd_rate": self.illegal / self.locked if self.locked else 0.0,
        }
