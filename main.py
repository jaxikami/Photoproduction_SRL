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
STATE_DIM = 11     # [Cx, CN, Cq, V, stage_0..3, credit, t_norm, supply]
ACTION_DIM = 4     # [time_mult, I, Fn, Fout]
MAX_EPISODES = 50000
UPDATE_TIMESTEP = 1000
K_EPOCHS = 3
EPS_CLIP = 0.2
GAMMA = 0.95
LR_ACTOR = 5e-5
LR_CRITIC = 1e-4
MIN_LR = 1e-5
INITIAL_ENTROPY = 0.05
MIN_ENTROPY = 1e-5
EVALUATE_ONLY = False
RUN_BENCHMARK = True   # True → run Standard RL (bench) only; False → run Safe RL only
NOISE_STD = 0.1

class Memory:
    """Buffer for storing environment trajectories."""
    def __init__(self):
        self.states, self.actions, self.raw_actions, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], [], []

    def clear(self):
        del self.states[:], self.actions[:], self.raw_actions[:], self.logprobs[:], self.rewards[:], self.is_terminals[:]

def train_agent(agent_name, agent, logger):
    """Primary training loop for a specified RL agent."""
    print(f"\n--- Starting Training: {agent_name} ---")
    env = PhycocyaninEnvSafe() if agent_name == "Safe RL" else PhycocyaninEnvBench()
    memory = Memory()
    scheduler = LinearLR(agent.optimizer, start_factor=1.0, end_factor=MIN_LR / LR_ACTOR, total_iters=MAX_EPISODES)

    time_step = 0
    WINDOW_SIZE = 200
    EARLY_STOP_WARMUP = 5000
    EARLY_STOP_PATIENCE = 3000
    min_improvement = 1e-4
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

            memory.states.append(torch.tensor(state, dtype=torch.float32))
            memory.actions.append(torch.tensor(action, dtype=torch.float32))
            memory.raw_actions.append(torch.tensor(raw_act, dtype=torch.float32))
            memory.logprobs.append(torch.tensor(log_prob, dtype=torch.float32))

            state, reward, done, info = env.step(action)

            current_ep_reward += reward
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

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
            else:
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
                "TotR": f"{info['total_reward']:.1f}",
                "AvgR": f"{info['avg_reward']:.3f}",
                "VioID": vio_id_str,
                "g1P": f"{info['avg_g1_penalty']:.2f}",
                "g2P": f"{info['avg_g2_penalty']:.2f}",
                "g3P": f"{info['avg_g3_penalty']:.2f}",
                "g4P": f"{info['avg_g4_penalty']:.2f}",
                "g5P": f"{info['avg_g5_penalty']:.2f}",
                "g6P": f"{info['avg_g6_penalty']:.2f}",
                "SmthP": f"{info['avg_smooth_penalty']:.3f}",
                "RawMP": f"{info['avg_raw_mat_penalty']:.3f}",
                "TotVio": f"{info['violation_count']}",
                "Stage": f"{info['current_stage']}"
            })

    os.makedirs("policy", exist_ok=True)
    torch.save(agent.policy.state_dict(), os.path.join("policy", f"{agent_name}_final_weights.pth"))
    Plotter.plot_training_results(logger.training_log, agent_name=agent_name)

def evaluate_agent(agent_name, agent, logger, eval_episodes=10000, noise_std=0.05):
    """
    Evaluates a trained agent with randomized initial states and intent noise.
    """
    print(f"\n--- Evaluating: {agent_name} with N(0, {noise_std}) noise ---")

    load_path = os.path.join("policy", f"{agent_name}_final_weights.pth")
    if os.path.exists(load_path):
        agent.policy.load_state_dict(torch.load(load_path))
        agent.policy.eval()

    env = PhycocyaninEnvSafe() if agent_name == "Safe RL" else PhycocyaninEnvBench()
    total_g1, total_g2, total_g3, total_g4 = 0, 0, 0, 0

    all_episodes = []
    all_nitrate_trajectories    = []
    all_production_trajectories = []
    all_ratio_trajectories      = []
    all_volume_trajectories     = []

    for _ in tqdm(range(eval_episodes), desc=f"Evaluating {agent_name}"):
        state = env.reset(randomize=True)

        ep_states, ep_actions, ep_rewards, ep_infos = [], [], [], []
        eval_hidden = None

        while True:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
                if agent_name == "Safe RL":
                    z, _, _, eval_hidden = agent.policy.act(state_t, eval_hidden)
                else:
                    z, _, _ = agent.policy.act(state_t)
                intent = z.cpu().numpy().flatten()

            if noise_std > 0:
                noise = np.random.normal(0, noise_std, size=intent.shape)
                noisy_intent = np.clip(intent + noise, -1.0, 1.0)
            else:
                noisy_intent = intent

            with torch.no_grad():
                if agent_name == "Safe RL":
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

        if len(ep_infos) > 0:
            last_info = ep_infos[-1]
            logger.log_evaluation_episode_violations(
                agent_name,
                last_info["violation_count"],
                last_info.get("g1_violation_count", 0),
                last_info.get("g2_violation_count", 0),
                last_info.get("g3_violation_count", 0),
                last_info.get("g4_violation_count", 0)
            )
            total_g1 += last_info.get("g1_violation_count", 0)
            total_g2 += last_info.get("g2_violation_count", 0)
            total_g3 += last_info.get("g3_violation_count", 0)
            total_g4 += last_info.get("g4_violation_count", 0)

    print(f"{agent_name} Violations — G1: {total_g1}, G2: {total_g2}, "
          f"G3: {total_g3}, G4: {total_g4}")

    all_episodes.sort(key=lambda x: x[0])
    mid_idx = len(all_episodes) // 2
    median_ep_reward, median_states, median_actions, median_rewards, median_infos = all_episodes[mid_idx]

    for i in median_infos: i["is_safe"] = 1 if i["violation_count"] == 0 else 0

    all_nitrate = np.array(all_nitrate_trajectories)
    all_prod    = np.array(all_production_trajectories)
    all_ratio   = np.array(all_ratio_trajectories)
    all_vol     = np.array(all_volume_trajectories)

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
    }

    logger.log_evaluation_trajectory(agent_name, median_states, median_actions,
                                      median_rewards, median_infos, agg_data)
    print(f"Median Episode Reward: {median_ep_reward:.2f}")
    print(f"Mean Episode Reward: {np.mean([e[0] for e in all_episodes]):.2f}  |  "
          f"Std: {np.std([e[0] for e in all_episodes]):.2f}")

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
        agent_name = "Safe RL"

    if not EVALUATE_ONLY:
        train_agent(agent_name, agent, logger)
        if not EVALUATE_ONLY:
            Plotter.plot_training_violations(logger)

    evaluate_agent(agent_name, agent, logger, noise_std=NOISE_STD)
    Plotter.plot_comprehensive_evaluation(logger.eval_data, logger.eval_violations)