"""Noisy evaluation of a trained policy and the data behind its plots."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
from tqdm import tqdm

# ─── Local ────────────────────────────────────────────────────────────────────
from agents.checkpoint import load_checkpoint


def evaluate_agent(cfg, agent, logger, eval_episodes=1000):
    """Evaluates a trained agent with randomized initial states and injected noise.

    Runs the trained policy over a large number of episodes to establish statistical
    confidence in its safety and performance. Tracks trajectories, constraint
    violations, and calculates aggregate evaluation metrics.  Action noise is
    injected into the agent's intent before its safety stack, so the Safe RL
    agent's projection sees (and may correct) the perturbed action.

    Args:
        cfg (RunConfig): The run's settings (agent, noise settings, paths).
        agent: The agent built by cfg.make_agent().
        logger (DataLogger): The logging utility for evaluation metrics.
        eval_episodes (int, optional): Number of episodes to run. Defaults to 1000.
    """
    agent_name = cfg.label
    noise_std, action_noise, state_noise = cfg.noise_std, cfg.action_noise, cfg.state_noise
    if noise_std <= 0 or (not state_noise and not action_noise):
        noise_mode = "no noise"
    elif state_noise and action_noise:
        noise_mode = "state + action noise"
    elif state_noise:
        noise_mode = "state noise"
    else:
        noise_mode = "action noise"
    effective_pct = 0.0 if noise_mode == "no noise" else noise_std * 100.0

    print(
        f"\n--- Evaluating: {agent_name} with {noise_mode} "
        f"(relative +- {effective_pct:.1f}%, configured {noise_std * 100:.1f}%) ---"
    )

    if os.path.exists(cfg.model_path):
        load_checkpoint(agent, cfg.model_path)
    intent_noise = noise_std if action_noise else 0.0

    env = cfg.make_env()
    total_g = {k: 0 for k in range(1, 6)}

    all_episodes = []
    all_nitrate_trajectories    = []
    all_production_trajectories = []
    all_ratio_trajectories      = []
    all_volume_trajectories     = []
    all_harvested_trajectories  = []

    for _ in tqdm(range(eval_episodes), desc=f"Evaluating {agent_name}"):
        state = env.reset(randomize=True)
        agent.reset()

        ep_states, ep_actions, ep_rewards, ep_infos = [], [], [], []

        while True:
            if state_noise and noise_std > 0:
                s_noise = state * np.random.normal(0, noise_std, size=state.shape)
                noisy_state = np.clip(state + s_noise, 0.0, 1.0)
            else:
                noisy_state = state

            action, _, _, _ = agent.select_action(noisy_state, intent_noise=intent_noise)

            next_state, reward, done, info = env.step(action)
            ep_states.append(state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_infos.append(info)
            state = next_state
            if done:
                ep_states.append(state)
                break
        agent.discard_rollout()

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
            g_counts = [last_info.get(f"g{k}_violation_count", 0) for k in range(1, 6)]
            logger.log_evaluation_episode_violations(agent_name, last_info["violation_count"], *g_counts)
            for k, c in zip(range(1, 6), g_counts):
                total_g[k] += 1 if c > 0 else 0

    print(f"{agent_name} Violations — " + ", ".join(f"G{k}: {total_g[k]}" for k in range(1, 6)))

    all_episodes.sort(key=lambda x: x[0])
    mid_idx = len(all_episodes) // 2
    median_ep_reward, median_states, median_actions, median_rewards, median_infos = all_episodes[mid_idx]

    for i in median_infos:
        i["is_safe"] = 1 if i["violation_count"] == 0 else 0

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
    stages = np.array([info["current_stage"] for info in median_infos])
    # Match the stages array to the 101 elements of states (append last stage)
    stages_101 = np.append(stages, stages[-1])
    os.makedirs(os.path.dirname(cfg.eval_path), exist_ok=True)
    np.savez(cfg.eval_path,
             states=np.array(median_states),
             actions=np.array(median_actions),
             stages=stages_101,
             rewards=np.array(median_rewards),
             final_mass=agg_data["harvested_avg"][-1],
             final_conc=agg_data["production_avg"][-1] * 0.2)
    print(f"Saved evaluation trajectory to {cfg.eval_path}")
