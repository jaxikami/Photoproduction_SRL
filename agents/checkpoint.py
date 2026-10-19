"""Saving and loading a trained agent's policy.

A checkpoint is a dict {"policy": state_dict}.  A file holding a bare policy
state_dict (written before this format) still loads.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import torch

# ─── Local ────────────────────────────────────────────────────────────────────
from agents.networks import device


def save_checkpoint(agent, path):
    """Save the agent's policy to ``path``."""
    torch.save({"policy": agent.policy.state_dict()}, path)


def load_checkpoint(agent, path):
    """Load a checkpoint into ``agent`` (policy and policy_old).

    Args:
        agent: Agent built with the architecture the checkpoint was saved from.
        path: File written by save_checkpoint(), or a bare policy state_dict.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    if "policy" not in state:  # bare policy state_dict (older checkpoints)
        state = {"policy": state}
    agent.policy.load_state_dict(state["policy"])
    agent.policy_old.load_state_dict(state["policy"])
