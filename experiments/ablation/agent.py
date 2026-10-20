"""The ablated Safe RL agent, the Standard RL baseline and the study's environments."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

# ─── Local ────────────────────────────────────────────────────────────────────
from agents.bench import PPO_Agent as StandardRL_Agent
from agents.checkpoint import load_checkpoint
from agents.safe import PPO_Agent as SafePPO_Agent
from envs.bench import PhycocyaninEnvBench
from envs.safe import PhycocyaninEnvSafe
from experiments.ablation.config import ALL_CONFIGS, BASELINE
from experiments.config import (ACTION_DIM, EPS_CLIP, GAMMA, INITIAL_ENTROPY, K_EPOCHS, LR_ACTOR,
                                LR_CRITIC, STATE_DIM)
from safety.constants import KINETIC_PATH


class MemorylessGRU(nn.Module):
    """nn.GRU wrapper that starts every timestep from a zero hidden state.

    Wraps the policy's own GRU, so parameters, initial weights and optimizer
    groups are unchanged; the only difference from the full model is that no
    information flows between timesteps.  The incoming hidden state is returned
    as-is (it stays at zero), so callers that thread h through are unaffected.

    Attributes:
        gru: The wrapped nn.GRU (batch_first).
    """

    def __init__(self, gru):
        """Wrap an existing GRU.

        Args:
            gru: batch_first nn.GRU to strip of its memory.
        """
        super().__init__()
        self.gru = gru

    def forward(self, x, h):
        """Run each timestep of x as an independent length-1 sequence.

        Args:
            x: Input of shape [B, T, input_size].
            h: Hidden state from the caller; ignored and returned unchanged.

        Returns:
            A tuple of (output of shape [B, T, hidden_size], h).
        """
        B, T, D = x.shape
        h0 = torch.zeros(self.gru.num_layers, B * T, self.gru.hidden_size,
                         device=x.device, dtype=x.dtype)
        out, _ = self.gru(x.reshape(B * T, 1, D), h0)
        return out.reshape(B, T, -1), h


class AblationAgent(SafePPO_Agent):
    """Safe RL agent whose GRU memory, stage mask and APN stages can be switched off.

    The Lagrangian lives in the environment and is toggled by make_env().
    """

    def __init__(self, cfg):
        """Build the Safe RL agent with the ablated components removed.

        Args:
            cfg: AblationConfig selecting the active components.

        Raises:
            FileNotFoundError: If the G3 head is needed but KineticNet's
                checkpoint is missing (the parent would silently leave G3
                uncovered).
        """
        if (cfg.proj or cfg.grad) and cfg.g3 and not os.path.exists(KINETIC_PATH):
            raise FileNotFoundError(f"{KINETIC_PATH} not found; run python -m safety.train first.")
        super().__init__(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP,
                         INITIAL_ENTROPY, use_safeguard=True, use_projection=cfg.proj,
                         use_safety_grad=cfg.grad, use_g3=cfg.g3)
        if not cfg.gru:
            self.policy.gru = MemorylessGRU(self.policy.gru)
            self.policy_old.gru = MemorylessGRU(self.policy_old.gru)
        if not cfg.mask:
            # The whole of L0: the policy distribution is unmasked and the
            # interlock no longer overwrites locked dimensions.  The nominal
            # plant ignores a locked actuator anyway; under the 'actuators'
            # scenario it runs it (a G7 breach).
            self.policy.use_safeguard = False
            self.policy_old.use_safeguard = False
            self.use_interlock = False


def make_agent(name):
    """A fresh agent for a run: an ablation config's AblationAgent or the baseline."""
    if name == BASELINE:
        return StandardRL_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS,
                                EPS_CLIP, INITIAL_ENTROPY)
    return AblationAgent(ALL_CONFIGS[name])


def make_train_env(name, scenarios=()):
    """The environment a run trains on (the baseline trains with static penalties)."""
    if name == BASELINE:
        return PhycocyaninEnvBench(scenarios)
    return make_env(ALL_CONFIGS[name].lagrangian, scenarios)


def make_env(use_lagrangian, scenarios=()):
    """Create the PhycocyaninEnvSafe the study trains and evaluates on.

    Args:
        use_lagrangian: If False, dual ascent is disabled; the multipliers stay
            at 0, leaving only the static constraint shaping.
        scenarios: envs/plant.SCENARIOS to switch on.

    Returns:
        PhycocyaninEnvSafe instance.
    """
    env = PhycocyaninEnvSafe(scenarios)
    if not use_lagrangian:
        env.disable_lagrangian()
    return env


def load_agent(name, run_dir):
    """Rebuild a trained run from its saved policy.pth.

    Args:
        name: Ablation config name or BASELINE.
        run_dir: Run directory holding policy.pth.

    Returns:
        The agent with the trained policy in both policy and policy_old.
    """
    agent = make_agent(name)
    load_checkpoint(agent, run_dir / "policy.pth")
    return agent
