"""Output folders of the generated plots."""
import os

PLOT_DIR = "plot"
PLOT_BENCH_DIR = os.path.join(PLOT_DIR, "plot_bench")      # Standard RL (benchmark) agent
PLOT_SAFE_DIR = os.path.join(PLOT_DIR, "plot_safe")        # Safe RL agent and APN pretraining
PLOT_COMPARE_DIR = os.path.join(PLOT_DIR, "plot_compare")  # both agents side by side


def agent_plot_dir(agent_name):
    """Plot folder of one agent, created if missing."""
    out_dir = PLOT_SAFE_DIR if "safe" in agent_name.lower() else PLOT_BENCH_DIR
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def compare_plot_dir():
    """Plot folder of the two-agent comparisons, created if missing."""
    os.makedirs(PLOT_COMPARE_DIR, exist_ok=True)
    return PLOT_COMPARE_DIR
