"""Study settings: training protocol, constraints, noise and component configs."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import itertools
from dataclasses import dataclass

# ─── Study Defaults ───────────────────────────────────────────────────────────

# main.MAX_EPISODES.  The LR and entropy schedules span this many episodes, so
# a shorter ablation budget follows the main model's schedule instead of a
# compressed one.
SCHEDULE_EPISODES = 20000

DEFAULT_EPISODES = 10000
DEFAULT_EVAL_EPISODES = 100
DEFAULT_SEEDS = [0, 1, 2]
# Not measured for this project: rollouts are one reactor at a time
# (latency-bound), so several processes share the GPU; lower it if they contend.
DEFAULT_WORKERS = 6

# Evaluation reseeds with seed + offset so its sampling noise does not depend
# on how training consumed the RNG.
EVAL_SEED_OFFSET = 10_000

# Bumped whenever a change invalidates trained runs (plant, constraints or
# agent); a run trained under another version is retrained, not re-evaluated.
STUDY_VERSION = 1

# A raw command on a stage-locked dimension counts as illegal when it differs
# from the interlock default by more than this, in [-1, 1] action units.
ILLEGAL_CMD_TOL = 0.05

# ─── Constraints ──────────────────────────────────────────────────────────────

# Constraint ids as PhycocyaninEnvSafe counts them (info["g<k>_violation_count"]).
VIOLATION_NAMES = {1: "path_nitrate", 2: "product_ratio", 3: "terminal_nitrate",
                   4: "overflow", 5: "end_idle"}
VIOLATION_LABELS = {"path_nitrate": "G1 path nitrate", "product_ratio": "G2 Cq/Cx ratio",
                    "terminal_nitrate": "G3 harvest-start nitrate", "overflow": "G4 overflow",
                    "end_idle": "G5 end in Idle"}

# Violations with a magnitude (G5 is pass/fail).
SEVERITY_TYPES = ("path_nitrate", "product_ratio", "terminal_nitrate", "overflow")
SEVERITY_UNITS = {"path_nitrate": "% over 800 mg/L", "product_ratio": "% over 0.011",
                  "terminal_nitrate": "% over 150 mg/L", "overflow": "% over V_MAX"}

# ─── Evaluation Noise ─────────────────────────────────────────────────────────
NOISE_MODES = ("none", "action", "state")
NOISE_LABELS = {"none": "No noise", "action": "Action noise", "state": "State noise"}
DEFAULT_NOISE_STD = 0.05

# Action dims that drive physical actuators (light, nitrate feed, outflow).
# The time multiplier (0) is a scheduling decision, not an actuator.
FEED_DIMS = [1, 2, 3]
# Measured observation dims: Cx, CN, Cq, V and the nitrate supply level.
# Stage, credit and time are known exactly and stay clean.
STATE_NOISE_DIMS = [0, 1, 2, 3, 10]

# ─── Ablation Configurations ──────────────────────────────────────────────────

COMPONENTS = ("gru", "lagrangian", "mask", "proj", "grad")
COMPONENT_LABELS = {"gru": "GRU", "lagrangian": "Lagrangian", "mask": "mask",
                    "proj": "projection", "grad": "safety gradient"}


@dataclass(frozen=True)
class AblationConfig:
    """Which Safe RL components are active in one variant.

    Attributes:
        gru: Recurrent memory (hidden state carried across timesteps).
        lagrangian: Adaptive Lagrange multipliers in the environment.
        mask: L0, SERL stage mask applied to the policy distribution.
        proj: L2, APN projection of every action.
        grad: The APN's safety gradient in the PPO update.
    """
    gru: bool = True
    lagrangian: bool = True
    mask: bool = True
    proj: bool = True
    grad: bool = True

    @property
    def removed(self):
        """Components switched off in this variant, in COMPONENTS order."""
        return [c for c in COMPONENTS if not getattr(self, c)]

    @property
    def name(self):
        """Identifier used on the command line and in output paths."""
        if not self.removed:
            return "full"
        if len(self.removed) == len(COMPONENTS):
            return "none"
        return "no_" + "_".join(self.removed)

    @property
    def label(self):
        """Human-readable name for tables and plots."""
        if not self.removed:
            return "Full"
        if len(self.removed) == len(COMPONENTS):
            return "w/o all"
        return "w/o " + " + ".join(COMPONENT_LABELS[c] for c in self.removed)


# Every combination, ordered full -> single ablations -> pairs -> ... -> none.
ALL_CONFIGS = {
    cfg.name: cfg for cfg in sorted(
        (AblationConfig(*flags)
         for flags in itertools.product((True, False), repeat=len(COMPONENTS))),
        key=lambda c: (len(c.removed), [COMPONENTS.index(r) for r in c.removed]))
}
LOO_CONFIGS = ["full"] + [f"no_{c}" for c in COMPONENTS]
