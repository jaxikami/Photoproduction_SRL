"""Training and evaluation entry point for Photoproduction RL.

Trains and/or evaluates either the Standard RL (benchmark) PPO agent or the
Safe RL agent on the continuous photobioreactor.  Edit the run configuration
below, then:  python main.py
"""

import os

# Suppress library warnings BEFORE importing heavy libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from experiments.config import RunConfig  # noqa: E402
from experiments.evaluate import evaluate_agent  # noqa: E402
from experiments.train import train_agent  # noqa: E402
from reporting.logger import DataLogger  # noqa: E402
from reporting.plots import Plotter  # noqa: E402

# ---------------- Run configuration ----------------
EVALUATE_ONLY = True    # True → skip training and run evaluation only; False → run full train + eval
RUN_BENCHMARK = True    # True → run Standard RL (bench) only; False → run Safe RL only
RESUME_TRAINING = True  # True → load existing weights before training
MAX_EPISODES = 20000
NOISE_STD = 0.05
ACTION_NOISE = True
STATE_NOISE = False


if __name__ == "__main__":
    cfg = RunConfig(benchmark=RUN_BENCHMARK, max_episodes=MAX_EPISODES, resume=RESUME_TRAINING,
                    noise_std=NOISE_STD, action_noise=ACTION_NOISE, state_noise=STATE_NOISE)
    logger = DataLogger()
    agent = cfg.make_agent()

    if not EVALUATE_ONLY:
        train_agent(cfg, agent, logger)
        Plotter.plot_training_violations(logger)

    evaluate_agent(cfg, agent, logger)
    Plotter.plot_comprehensive_evaluation(logger, cfg.label)
    Plotter.plot_comparison_table()
