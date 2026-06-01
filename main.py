"""
Main Training and Evaluation Script for Photoproduction RL.

This script coordinates the training and evaluation of either the Benchmark (Standard)
PPO agent or the Safe (SPRL) agent within the continuous photobioreactor environment.
It handles hyperparameter configuration, trajectory collection, early stopping,
and invoking the DataLogger and Plotter for visualization.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*")

import torch
import torch.nn as nn
import numpy as np
import os
from collections import deque
from tqdm import tqdm
from env_bench import PhycocyaninEnvBench
from env_safe import PhycocyaninEnvSafe
from lag_agent import StandardRL_Agent
from safe_agent import SPRL_Agent as SafeRL_Agent
from utils import DataLogger, Plotter
from torch.optim.lr_scheduler import LinearLR

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
STATE_DIM = 12     # [Cx, CN, Cq, V, stage_0..3, credit, t_norm, supply, op_time_left]
ACTION_DIM = 4     # [time_mult, I, Fn, Fout]
MAX_EPISODES = 50000
UPDATE_TIMESTEP = 2048
K_EPOCHS = 4
EPS_CLIP = 0.2
GAMMA = 0.95
LR_ACTOR = 5e-5
LR_CRITIC = 1e-4
MIN_LR = 1e-5
INITIAL_ENTROPY = 0.05
MIN_ENTROPY = 1e-5
EVALUATE_ONLY = False  # True → skip training and run evaluation only; False → run full train + eval
RUN_BENCHMARK = True   # True → run Standard RL (bench) only; False → run Safe RL only
RESUME_TRAINING = True # True → load existing weights before training
NOISE_STD = 0.05
ACTION_NOISE = True
STATE_NOISE = False

class Memory:
    """Buffer for storing environment trajectories during rollouts.
    
    Holds pre-allocated numpy arrays for states, actions, rewards, log-probabilities,
    and terminal flags to be consumed by the PPO optimization step.
    Avoids per-step tensor allocation overhead.
    """
    def __init__(self, capacity=UPDATE_TIMESTEP + 200, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        """Initializes pre-allocated numpy trajectory arrays."""
        self.capacity = capacity
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._raw_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._logprobs = np.zeros(capacity, dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._is_terminals = np.zeros(capacity, dtype=np.bool_)
        self._ptr = 0

    def push(self, state, action, raw_action, logprob, reward, is_terminal):
        """Stores one transition into pre-allocated arrays."""
        i = self._ptr
        self._states[i] = state
        self._actions[i] = action
        self._raw_actions[i] = raw_action
        self._logprobs[i] = np.asarray(logprob).item()
        self._rewards[i] = reward
        self._is_terminals[i] = is_terminal
        self._ptr += 1

    @property
    def states(self):
        return [torch.from_numpy(self._states[i]) for i in range(self._ptr)]

    @property
    def actions(self):
        return [torch.from_numpy(self._actions[i]) for i in range(self._ptr)]

    @property
    def raw_actions(self):
        return [torch.from_numpy(self._raw_actions[i]) for i in range(self._ptr)]

    @property
    def logprobs(self):
        return [torch.tensor(self._logprobs[i]) for i in range(self._ptr)]

    @property
    def rewards(self):
        return self._rewards[:self._ptr].tolist()

    @property
    def is_terminals(self):
        return self._is_terminals[:self._ptr].tolist()

    def get_tensors(self, device):
        """Returns all data as pre-stacked tensors on the given device (fast path)."""
        n = self._ptr
        states = torch.from_numpy(self._states[:n]).to(device)
        actions = torch.from_numpy(self._actions[:n]).to(device)
        raw_actions = torch.from_numpy(self._raw_actions[:n]).to(device)
        logprobs = torch.from_numpy(self._logprobs[:n]).to(device)
        is_terminals = torch.from_numpy(self._is_terminals[:n]).to(device)
        return states, actions, raw_actions, logprobs, is_terminals

    def clear(self):
        """Resets pointer to reuse pre-allocated memory."""
        self._ptr = 0

def train_agent(agent_name, agent, logger):
    """Primary training loop for a specified RL agent.

    Iteratively collects trajectories using the environment, triggers the agent's
    PPO optimization subroutine, tracks constraint violations, handles early
    stopping heuristics, and logs performance metrics.

    Args:
        agent_name (str): Identifier for the agent (e.g., "Standard RL" or "safe_RL agent").
        agent (object): The instantiated RL agent object (StandardRL_Agent or SPRL_Agent).
        logger (DataLogger): The logging utility for tracking metrics.
    """
    print(f"\n--- Starting Training: {agent_name} ---")
    if RESUME_TRAINING:
        load_path = os.path.join("policy", f"{agent_name}_final_weights.pth")
        if os.path.exists(load_path):
            print(f"Resuming training from {load_path}")
            agent.policy.load_state_dict(torch.load(load_path))
        else:
            print(f"Warning: {load_path} not found. Starting from scratch.")

    env = PhycocyaninEnvSafe() if agent_name == "safe_RL agent" else PhycocyaninEnvBench()
    memory = Memory()
    scheduler = LinearLR(agent.optimizer, start_factor=1.0, end_factor=MIN_LR / LR_ACTOR, total_iters=MAX_EPISODES)

    time_step = 0
    WINDOW_SIZE = 200
    EARLY_STOP_WARMUP = 15000
    EARLY_STOP_PATIENCE = 3000
    min_improvement = 1e-3 if agent_name == "Standard RL" else 1e-4
    rewards_window = deque(maxlen=WINDOW_SIZE)
    best_avg_reward = -float('inf')
    no_improve_count = 0

    pbar = tqdm(range(1, MAX_EPISODES + 1), desc=f"Training {agent_name}")
    for i_episode in pbar:
        state = env.reset()
        current_ep_reward = 0

        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()
        elif hasattr(agent, 'reset'):
            agent.reset()

        while True:
            time_step += 1

            action, log_prob, raw_act = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            memory.push(state, action, raw_act, log_prob, reward, done)
            current_ep_reward += reward
            state = next_state

            if time_step % UPDATE_TIMESTEP == 0:
                agent.learn(memory)
                memory.clear()
                time_step = 0

            if done: break

        # Anneal entropy coefficient proportionally to remaining LR headroom.
        curr_lr = agent.optimizer.param_groups[0]['lr']
        cos_frac = (curr_lr - MIN_LR) / (LR_ACTOR - MIN_LR + 1e-12)
        agent.entropy_coeff = MIN_ENTROPY + (INITIAL_ENTROPY - MIN_ENTROPY) * cos_frac

        scheduler.step()
        logger.log_training_episode(agent_name, current_ep_reward, info["violation_count"])
        rewards_window.append(current_ep_reward)

        if i_episode > EARLY_STOP_WARMUP and len(rewards_window) == WINDOW_SIZE:
            avg_reward = np.mean(rewards_window)
            if avg_reward > best_avg_reward * (1 + min_improvement):
                best_avg_reward = avg_reward
                no_improve_count = 0
                os.makedirs("policy", exist_ok=True)
                torch.save(agent.policy.state_dict(),
                           os.path.join("policy", f"{agent_name}_best_weights.pth"))
            elif i_episode > 15000:
                no_improve_count += 1

            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"[Early Stopping] No improvement for {EARLY_STOP_PATIENCE} episodes. "
                      f"Best avg reward: {best_avg_reward:.3f}")
                break

        if i_episode % 10 == 0:
            vio_id_str = "".join(
                str(k) for k, c in [
                    (1, info.get("g1_violation_count", 0)),
                    (2, info.get("g2_violation_count", 0)),
                    (3, info.get("g3_violation_count", 0)),
                    (4, info.get("g4_violation_count", 0)),
                    (5, info.get("g5_violation_count", 0)),
                ] if c > 0
            ) or "-"
            pbar.set_postfix({
                "TotR": f"{info['total_reward'] / 300.0:.3f}",
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
                "Stage": f"{info['current_stage']}"
            })

    os.makedirs("policy", exist_ok=True)
    torch.save(agent.policy.state_dict(), os.path.join("policy", f"{agent_name}_final_weights.pth"))
    Plotter.plot_training_results(logger.training_log, agent_name=agent_name)

def evaluate_agent(agent_name, agent, logger, eval_episodes=1000, noise_std=0.05, action_noise=False, state_noise=True):
    """Evaluates a trained agent with randomized initial states and injected intent noise.

    Runs the trained policy over a large number of episodes to establish statistical
    confidence in its safety and performance. Tracks trajectories, constraint
    violations, and calculates aggregate evaluation metrics.

    Args:
        agent_name (str): Identifier for the agent to evaluate.
        agent (object): The instantiated RL agent object.
        logger (DataLogger): The logging utility for evaluation metrics.
        eval_episodes (int, optional): Number of episodes to run. Defaults to 1000.
        noise_std (float, optional): Standard deviation of Gaussian noise. Defaults to 0.05.
        action_noise (bool, optional): Whether to inject noise into the agent's actions. Defaults to True.
        state_noise (bool, optional): Whether to inject noise into the environment state observations. Defaults to True.
    """
    print(f"\n--- Evaluating: {agent_name} with N(0, {noise_std}) noise ---")

    load_path = os.path.join("policy", f"{agent_name}_final_weights.pth")
    if os.path.exists(load_path):
        agent.policy.load_state_dict(torch.load(load_path))
        agent.policy.eval()

    env = PhycocyaninEnvSafe() if agent_name == "safe_RL agent" else PhycocyaninEnvBench()
    total_g1, total_g2, total_g3, total_g4, total_g5 = 0, 0, 0, 0, 0

    all_episodes = []
    all_nitrate_trajectories    = []
    all_production_trajectories = []
    all_ratio_trajectories      = []
    all_volume_trajectories     = []
    all_harvested_trajectories  = []

    for _ in tqdm(range(eval_episodes), desc=f"Evaluating {agent_name}"):
        state = env.reset(randomize=True)

        ep_states, ep_actions, ep_rewards, ep_infos = [], [], [], []
        eval_hidden = None

        while True:
            if state_noise and noise_std > 0:
                s_noise = np.random.normal(0, noise_std, size=state.shape)
                noisy_state = state + s_noise
            else:
                noisy_state = state

            with torch.no_grad():
                state_t = torch.FloatTensor(noisy_state).to(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
                if agent_name == "safe_RL agent":
                    z, _, _, eval_hidden, _, _, _ = agent.policy.act(state_t, eval_hidden)
                else:
                    z, _, _ = agent.policy.act(state_t)
                intent = z.cpu().numpy().flatten()

            if action_noise and noise_std > 0:
                noise = np.random.normal(0, noise_std, size=intent.shape)
                noisy_intent = np.clip(intent + noise, -1.0, 1.0)
            else:
                noisy_intent = intent

            with torch.no_grad():
                if agent_name == "safe_RL agent":
                    noisy_t = torch.FloatTensor(noisy_intent).to(state_t.device).unsqueeze(0)
                    # For Safe RL, apply the projection filter
                    from env_core import PhycocyaninEnvCore as _Env
                    mask = _Env.get_action_mask(state_t)
                    default_sq = torch.tensor([-0.333, -1.0, -1.0, -1.0], device=state_t.device)
                    noisy_t = noisy_t * mask + default_sq * (1 - mask)
                    action = agent._project_to_safe(state_t, noisy_t).cpu().numpy().flatten()
                    action_t = torch.FloatTensor(action).to(state_t.device).unsqueeze(0)
                    action = (action_t * mask + default_sq * (1 - mask)).cpu().numpy().flatten()
                else:
                    action = noisy_intent

            next_state, reward, done, info = env.step(action)
            ep_states.append(state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_infos.append(info)
            state = next_state
            if done:
                ep_states.append(state)
                break

        ep_total_reward = sum(ep_rewards)
        all_episodes.append((ep_total_reward, ep_states, ep_actions, ep_rewards, ep_infos))

        all_nitrate_trajectories.append([s[1] for s in ep_states])
        all_production_trajectories.append([s[2] for s in ep_states])
        all_ratio_trajectories.append(
            [(s[2] * 0.2) / (s[0] * 6.0 + 1e-8) for s in ep_states])
        all_volume_trajectories.append([s[3] for s in ep_states])
        # Track cumulative harvested mass (starts at 0.0, prepended)
        all_harvested_trajectories.append([0.0] + [info.get("total_cq_harvested", 0.0) for info in ep_infos])

        if len(ep_infos) > 0:
            last_info = ep_infos[-1]
            logger.log_evaluation_episode_violations(
                agent_name,
                last_info["violation_count"],
                last_info.get("g1_violation_count", 0),
                last_info.get("g2_violation_count", 0),
                last_info.get("g3_violation_count", 0),
                last_info.get("g4_violation_count", 0),
                last_info.get("g5_violation_count", 0)
            )
            total_g1 += 1 if last_info.get("g1_violation_count", 0) > 0 else 0
            total_g2 += 1 if last_info.get("g2_violation_count", 0) > 0 else 0
            total_g3 += 1 if last_info.get("g3_violation_count", 0) > 0 else 0
            total_g4 += 1 if last_info.get("g4_violation_count", 0) > 0 else 0
            total_g5 += 1 if last_info.get("g5_violation_count", 0) > 0 else 0

    print(f"{agent_name} Violations — G1: {total_g1}, G2: {total_g2}, "
          f"G3: {total_g3}, G4: {total_g4}, G5: {total_g5}")

    all_episodes.sort(key=lambda x: x[0])
    mid_idx = len(all_episodes) // 2
    median_ep_reward, median_states, median_actions, median_rewards, median_infos = all_episodes[mid_idx]

    for i in median_infos: i["is_safe"] = 1 if i["violation_count"] == 0 else 0

    all_nitrate = np.array(all_nitrate_trajectories)
    all_prod    = np.array(all_production_trajectories)
    all_ratio   = np.array(all_ratio_trajectories)
    all_vol     = np.array(all_volume_trajectories)
    all_harv    = np.array(all_harvested_trajectories)

    agg_data = {
        "nitrate_min":    np.min(all_nitrate, axis=0),
        "nitrate_max":    np.max(all_nitrate, axis=0),
        "production_avg": np.mean(all_prod, axis=0),
        "ratio_min":      np.min(all_ratio, axis=0),
        "ratio_max":      np.max(all_ratio, axis=0),
        "ratio_avg":      np.mean(all_ratio, axis=0),
        "ratio_std":      np.std(all_ratio, axis=0),
        "volume_min":     np.min(all_vol, axis=0),
        "volume_max":     np.max(all_vol, axis=0),
        "volume_avg":     np.mean(all_vol, axis=0),
        "harvested_avg":  np.mean(all_harv, axis=0),
    }

    logger.log_evaluation_trajectory(agent_name, median_states, median_actions,
                                      median_rewards, median_infos, agg_data)
    print(f"Median Episode Reward: {median_ep_reward:.2f}")
    print(f"Mean Episode Reward: {np.mean([e[0] for e in all_episodes]):.2f}  |  "
          f"Std: {np.std([e[0] for e in all_episodes]):.2f}")

    # Save evaluation data to npz for comparative plotting (e.g. Gantt charts)
    suffix = "safe" if "safe" in agent_name.lower() else "standard"
    npz_path = f"eval_data_{suffix}.npz"
    stages = np.array([info["current_stage"] for info in median_infos])
    # Match the stages array to the 101 elements of states (append last stage)
    stages_101 = np.append(stages, stages[-1])
    np.savez(npz_path, 
             states=np.array(median_states), 
             actions=np.array(median_actions), 
             stages=stages_101, 
             rewards=np.array(median_rewards))
    print(f"Saved evaluation trajectory to {npz_path}")

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    logger = DataLogger()

    if RUN_BENCHMARK:
        agent      = StandardRL_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC,
                                      GAMMA, K_EPOCHS, EPS_CLIP, INITIAL_ENTROPY)
        agent_name = "Standard RL"
    else:
        agent      = SafeRL_Agent(STATE_DIM, ACTION_DIM, LR_ACTOR, LR_CRITIC,
                                   GAMMA, K_EPOCHS, EPS_CLIP, INITIAL_ENTROPY)
        agent_name = "safe_RL agent"

    if not EVALUATE_ONLY:
        train_agent(agent_name, agent, logger)
        if not EVALUATE_ONLY:
            Plotter.plot_training_violations(logger)

    evaluate_agent(agent_name, agent, logger, noise_std=NOISE_STD, action_noise=ACTION_NOISE, state_noise=STATE_NOISE)
    Plotter.plot_comprehensive_evaluation(logger, agent_name)