"""Run settings of main.py: which agent, its PPO hyperparameters, and where its outputs go."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
from dataclasses import dataclass

# ─── Local ────────────────────────────────────────────────────────────────────
from agents.bench import PPO_Agent as StandardRL_Agent
from agents.safe import PPO_Agent as SafeRL_Agent
from envs.bench import PhycocyaninEnvBench
from envs.safe import PhycocyaninEnvSafe

# ─── PPO Hyperparameters ──────────────────────────────────────────────────────
STATE_DIM = 12     # [Cx, CN, Cq, V, stage_0..3, credit, t_norm, supply, op_time_left]
ACTION_DIM = 4     # [time_mult, I, Fn, Fout]
UPDATE_TIMESTEP = 2048
K_EPOCHS = 4
EPS_CLIP = 0.2
GAMMA = 0.95
LR_ACTOR = 5e-5
LR_CRITIC = 1e-4
MIN_LR = 1e-5
INITIAL_ENTROPY = 0.05
MIN_ENTROPY = 1e-5


@dataclass(frozen=True)
class RunConfig:
    """One main.py run.

    Attributes:
        benchmark: Standard RL on PhycocyaninEnvBench; else the Safe RL agent on
            PhycocyaninEnvSafe with its full safety stack.
        max_episodes: Training episodes (the LR and entropy schedules span them).
        resume: Load the existing final checkpoint before training.
        noise_std: Relative evaluation noise (0.05 = 5%).
        action_noise: Inject noise into the agent's action intent.
        state_noise: Inject noise into the observations.
    """
    benchmark: bool
    max_episodes: int = 20000
    resume: bool = True
    noise_std: float = 0.05
    action_noise: bool = True
    state_noise: bool = False

    @property
    def label(self):
        """Agent name used for logs, plots and checkpoint file names."""
        return "Standard RL" if self.benchmark else "safe_RL agent"

    @property
    def suffix(self):
        """Short name of the agent in evaluation file names."""
        return "standard" if self.benchmark else "safe"

    @property
    def model_path(self):
        """Final checkpoint."""
        return os.path.join("policy", f"{self.label}_final_weights.pth")

    @property
    def best_path(self):
        """Best-so-far checkpoint saved during training."""
        return os.path.join("policy", f"{self.label}_best_weights.pth")

    @property
    def eval_path(self):
        """Median evaluation trajectory (for the comparison table and Gantt chart)."""
        return os.path.join("policy", f"eval_data_{self.suffix}.npz")

    def make_env(self):
        """The environment this run's agent trains and is evaluated on."""
        return PhycocyaninEnvBench() if self.benchmark else PhycocyaninEnvSafe()

    def make_agent(self):
        """This run's agent with the PPO hyperparameters above."""
        agent_cls = StandardRL_Agent if self.benchmark else SafeRL_Agent
        return agent_cls(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS,
                         EPS_CLIP, INITIAL_ENTROPY)
