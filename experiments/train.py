"""The PPO training loop of main.py (Standard RL or Safe RL)."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
import warnings
from collections import deque

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

# ─── Local ────────────────────────────────────────────────────────────────────
from agents.checkpoint import load_checkpoint, save_checkpoint
from agents.memory import PPOMemory
from experiments.config import (ACTION_DIM, INITIAL_ENTROPY, LR_ACTOR, MIN_ENTROPY, MIN_LR,
                                STATE_DIM, UPDATE_TIMESTEP)
from reporting.plots import Plotter

warnings.filterwarnings("ignore", category=UserWarning,
                        message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*")


def train_agent(cfg, agent, logger):
    """Primary training loop for one RL agent.

    Iteratively collects trajectories using the environment, triggers the agent's
    PPO optimization subroutine, tracks constraint violations, handles early
    stopping heuristics, and logs performance metrics.

    Args:
        cfg (RunConfig): The run's settings.
        agent: The agent built by cfg.make_agent().
        logger (DataLogger): The logging utility for tracking metrics.
    """
    agent_name = cfg.label
    print(f"\n--- Starting Training: {agent_name} ---")
    if cfg.resume:
        if os.path.exists(cfg.model_path):
            print(f"Resuming training from {cfg.model_path}")
            load_checkpoint(agent, cfg.model_path)
        else:
            print(f"Warning: {cfg.model_path} not found. Starting from scratch.")

    env = cfg.make_env()
    memory = PPOMemory(UPDATE_TIMESTEP, STATE_DIM, ACTION_DIM)
    scheduler = LinearLR(agent.optimizer, start_factor=1.0, end_factor=MIN_LR / LR_ACTOR,
                         total_iters=cfg.max_episodes)

    time_step = 0
    WINDOW_SIZE = 200
    EARLY_STOP_WARMUP = 10000
    EARLY_STOP_PATIENCE = 2000
    min_improvement = 1e-3
    rewards_window = deque(maxlen=WINDOW_SIZE)
    best_avg_reward = -float('inf')
    no_improve_count = 0
    os.makedirs("policy", exist_ok=True)

    pbar = tqdm(range(1, cfg.max_episodes + 1), desc=f"Training {agent_name}")
    for i_episode in pbar:
        state = env.reset()
        agent.reset()  # Clear recurrent state at the start of each episode
        current_ep_reward = 0

        while True:
            time_step += 1

            action, raw_intent, log_prob, state_in = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            memory.append(state_in, raw_intent, log_prob, action, reward, done)
            current_ep_reward += reward
            state = next_state

            if time_step % UPDATE_TIMESTEP == 0:
                agent.learn(memory)
                memory.clear()
                time_step = 0

            if done:
                break

        # Anneal entropy coefficient proportionally to remaining LR headroom.
        curr_lr = agent.optimizer.param_groups[0]['lr']
        cos_frac = (curr_lr - MIN_LR) / (LR_ACTOR - MIN_LR + 1e-12)
        agent.entropy_coeff = MIN_ENTROPY + (INITIAL_ENTROPY - MIN_ENTROPY) * cos_frac

        scheduler.step()
        logger.log_training_episode(agent_name, current_ep_reward, info["violation_count"])
        rewards_window.append(current_ep_reward)

        if i_episode > EARLY_STOP_WARMUP and len(rewards_window) == WINDOW_SIZE:
            avg_reward = np.mean(rewards_window)
            if avg_reward > best_avg_reward + abs(best_avg_reward) * min_improvement:
                best_avg_reward = avg_reward
                no_improve_count = 0
                save_checkpoint(agent, cfg.best_path)
            else:
                no_improve_count += 1

            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"[Early Stopping] No improvement for {EARLY_STOP_PATIENCE} episodes. "
                      f"Best avg reward: {best_avg_reward:.3f}")
                break

        if i_episode % 10 == 0:
            vio_id_str = "".join(
                str(k) for k in range(1, 6) if info.get(f"g{k}_violation_count", 0) > 0
            ) or "-"
            postfix = {
                "AvgR": f"{info['avg_reward']:.3f}",
                "VioID": vio_id_str,
                "ProdR": f"{info['avg_prod_reward']:.2f}",
                "HarvR": f"{info['avg_harvest_reward']:.2f}",
                "g1P": f"{info['avg_g1_penalty']:.2f}",
                "g2P": f"{info['avg_g2_penalty']:.2f}",
                "g3P": f"{info['avg_g3_penalty']:.2f}",
                "g4P": f"{info['avg_g4_penalty']:.2f}",
                "g5P": f"{info['avg_g5_penalty']:.2f}",
                "SmthP": f"{info['avg_smooth_penalty']:.3f}",
                "RawMP": f"{info['avg_raw_mat_penalty']:.3f}",
                "TotVio": f"{info['violation_count']}",
                "Stage": f"{info['current_stage']}",
            }
            # APN projection diagnostics (Safe RL only): % of steps that were
            # already safe, and mean ascent steps when projection was needed.
            if hasattr(agent, 'get_and_reset_proj_stats'):
                proj = agent.get_and_reset_proj_stats()
                postfix["P_noop%"] = f"{100 * proj['noop'] / max(proj['calls'], 1):.0f}"
                postfix["P_avgIt"] = f"{proj['avg_it']:.1f}"
            pbar.set_postfix(postfix)

    save_checkpoint(agent, cfg.model_path)
    Plotter.plot_training_results(logger.training_log, agent_name=agent_name)
