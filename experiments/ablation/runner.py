"""Training and evaluation of one (config, seed) run, and the job pool that runs them."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import csv
import json
import multiprocessing
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

# ─── Local ────────────────────────────────────────────────────────────────────
from agents.checkpoint import save_checkpoint
from agents.memory import PPOMemory
from agents.networks import device
from experiments.ablation.agent import AblationAgent, load_agent, make_env
from experiments.ablation.config import (
    ALL_CONFIGS, COMPONENTS, DEFAULT_NOISE_STD, EVAL_SEED_OFFSET, FEED_DIMS, NOISE_MODES,
    SEVERITY_TYPES, STATE_NOISE_DIMS, STUDY_VERSION, VIOLATION_NAMES
)
from experiments.ablation.metrics import TRAIN_LOG_FIELDS, EpisodeStats
from experiments.config import (ACTION_DIM, INITIAL_ENTROPY, LR_ACTOR, MIN_ENTROPY, MIN_LR,
                                STATE_DIM, UPDATE_TIMESTEP)


def _seed_everything(seed):
    """Seed NumPy and torch (all devices)."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _noisy_state(state, std):
    """Apply multiplicative Gaussian sensor noise to the measured observation dims."""
    noisy = state.copy()
    noisy[STATE_NOISE_DIMS] *= 1.0 + np.random.normal(0.0, std, size=len(STATE_NOISE_DIMS))
    return np.maximum(noisy, 0.0)


def _noisy_action(action, std):
    """Apply multiplicative Gaussian actuator noise to the physical actuators.

    The noise scales the [0, 1] actuator ratio, so a closed valve stays closed.
    """
    noisy = action.copy()
    ratio = (noisy[FEED_DIMS] + 1.0) / 2.0
    ratio *= 1.0 + np.random.normal(0.0, std, size=ratio.shape)
    noisy[FEED_DIMS] = np.clip(ratio * 2.0 - 1.0, -1.0, 1.0)
    return noisy

# ─── Training and Evaluation ──────────────────────────────────────────────────


def train_run(cfg, seed, settings, run_dir):
    """Train one variant with experiments.train's protocol for a fixed budget.

    Args:
        cfg: AblationConfig of the variant.
        seed: Random seed for NumPy and torch.
        settings: Dict with 'episodes', 'schedule_episodes' and 'progress'.
        run_dir: Directory receiving train_log.csv and policy.pth.

    Returns:
        A tuple of (trained agent, dict of training statistics).
    """
    _seed_everything(seed)
    env = make_env(cfg.lagrangian)
    agent = AblationAgent(cfg)

    memory = PPOMemory(UPDATE_TIMESTEP, STATE_DIM, ACTION_DIM)
    scheduler = LinearLR(agent.optimizer, start_factor=1.0, end_factor=MIN_LR / LR_ACTOR,
                         total_iters=settings["schedule_episodes"])

    n_episodes = settings["episodes"]
    tag = f"{cfg.name} seed {seed}"
    pbar = tqdm(range(1, n_episodes + 1), desc=tag, disable=not settings["progress"])
    recent_harv = deque(maxlen=100)
    recent_viol = deque(maxlen=100)
    violations_total = 0
    global_step = 0
    t_start = time.perf_counter()

    with open(run_dir / "train_log.csv", "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=TRAIN_LOG_FIELDS)
        writer.writeheader()
        for episode in pbar:
            t_episode = time.perf_counter()
            stats = EpisodeStats()
            state = env.reset()
            agent.reset()
            done = False
            while not done:
                action, raw_intent, log_prob, state_in = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                memory.append(state_in, raw_intent, log_prob, action, reward, done)
                stats.record(env, state, raw_intent, action, reward, info)
                state = next_state

                global_step += 1
                if global_step % UPDATE_TIMESTEP == 0:
                    agent.learn(memory)
                    memory.clear()

            # Entropy annealing and LR schedule, as in experiments.train.
            lr_frac = ((agent.optimizer.param_groups[0]['lr'] - MIN_LR)
                       / (LR_ACTOR - MIN_LR + 1e-12))
            agent.entropy_coeff = MIN_ENTROPY + (INITIAL_ENTROPY - MIN_ENTROPY) * lr_frac
            scheduler.step()

            row = stats.finish(env, agent)
            row["episode"] = episode
            row["seconds"] = time.perf_counter() - t_episode
            writer.writerow(row)
            if episode % 100 == 0:
                log_file.flush()

            violations_total += row["violation_steps"]
            recent_harv.append(row["harvested"])
            recent_viol.append(row["violation_steps"])
            if settings["progress"]:
                pbar.set_postfix({'Harv': f'{np.mean(recent_harv):.2f}',
                                  'Viol': f'{np.mean(recent_viol):.2f}'})
            elif episode % 500 == 0:
                rate = (time.perf_counter() - t_start) / episode
                print(f"[{tag}] {episode:,}/{n_episodes:,} episodes | "
                      f"harvested {np.mean(recent_harv):.2f}/ep | "
                      f"viol {np.mean(recent_viol):.2f} steps/ep | {rate:.2f} s/ep", flush=True)

    elapsed = time.perf_counter() - t_start
    save_checkpoint(agent, run_dir / "policy.pth")
    return agent, {
        "hours": elapsed / 3600.0,
        "s_per_episode": elapsed / n_episodes,
        "violation_steps_total": int(violations_total),
        "harvested_last100": float(np.mean(recent_harv)),
        "violation_steps_last100": float(np.mean(recent_viol)),
        "policy_params": sum(p.numel() for p in agent.policy.parameters()),
    }


def evaluate_run(agent, seed, n_episodes, noise_mode="none", noise_std=DEFAULT_NOISE_STD):
    """Evaluate a trained agent under one noise mode with every Lagrange multiplier frozen at 0.

    A fresh environment keeps the multipliers learned during training out of
    the shaped return, so returns are comparable between variants trained with
    and without the Lagrangian.  Episodes start from randomized initial states
    and the policy samples stochastically, as in experiments.evaluate.

    Args:
        agent: Trained AblationAgent.
        seed: Run seed; evaluation reseeds with seed + EVAL_SEED_OFFSET.
        n_episodes: Number of evaluation episodes.
        noise_mode: One of NOISE_MODES.  'action' perturbs the actuators sent
            to the plant, 'state' perturbs the observation the agent acts on.
        noise_std: Relative standard deviation of the injected noise.

    Returns:
        Dict of evaluation metrics: means over episodes (plus selected stds),
        each constraint's events per episode and share of episodes violated,
        and the worst and mean-per-event exceedance of each SEVERITY_TYPES
        constraint over all episodes.
    """
    _seed_everything(seed + EVAL_SEED_OFFSET)
    env = make_env(use_lagrangian=False)
    agent.get_and_reset_proj_stats()  # drop counters left over from training

    rows, stats_all = [], []
    for _ in range(n_episodes):
        stats = EpisodeStats()
        state = env.reset(randomize=True)
        agent.reset()
        done = False
        while not done:
            obs = _noisy_state(state, noise_std) if noise_mode == "state" else state
            action, raw_intent, _, _ = agent.select_action(obs)
            sent = _noisy_action(action, noise_std) if noise_mode == "action" else action
            next_state, reward, done, info = env.step(sent)
            stats.record(env, obs, raw_intent, action, reward, info)
            state = next_state
        rows.append(stats.finish(env, agent))
        stats_all.append(stats)
        agent.discard_rollout()

    def values(key):
        return [r[key] for r in rows if r[key] is not None]

    def mean(key):
        vals = values(key)
        return float(np.mean(vals)) if vals else None

    def std(key):
        vals = values(key)
        return float(np.std(vals)) if vals else None

    noop = mean("apn_noop_frac")
    result = {
        "episodes": n_episodes,
        "noise_mode": noise_mode,
        "noise_std": 0.0 if noise_mode == "none" else noise_std,
        "return": mean("return"),
        "return_std": std("return"),
        "harvested": mean("harvested"),
        "harvested_std": std("harvested"),
        "violation_steps": mean("violation_steps"),
        "violation_steps_std": std("violation_steps"),
        "violation_episode_pct": 100.0 * float(np.mean([r["violation_steps"] > 0 for r in rows])),
        "ended_idle_pct": 100.0 * mean("ended_idle"),
        "illegal_cmd_rate": mean("illegal_cmd_rate"),
        "apn_intervention_rate": None if noop is None else 1.0 - noop,
        "apn_avg_iters": mean("apn_avg_iters"),
    }
    # Occurrence: events per episode and share of episodes with an event.
    for name in VIOLATION_NAMES.values():
        counts = [s.events[name] for s in stats_all]
        result[f"{name}_events"] = float(np.mean(counts))
        result[f"{name}_episode_pct"] = 100.0 * float(np.mean([c > 0 for c in counts]))
    # Severity: worst exceedance over all episodes, and mean exceedance per
    # event (None when the constraint never fired).
    for name in SEVERITY_TYPES:
        n_events = sum(s.events[name] for s in stats_all)
        result[f"{name}_peak_pct"] = 100.0 * max(s.peak[name] for s in stats_all)
        result[f"{name}_mean_depth_pct"] = (
            100.0 * sum(s.depth_sum[name] for s in stats_all) / n_events if n_events else None)
    result["peak_violation_pct"] = max(result[f"{n}_peak_pct"] for n in SEVERITY_TYPES)
    return result


def evaluate_all(agent, seed, n_episodes, noise_std, modes=NOISE_MODES):
    """Evaluate a trained agent under every noise mode (identically reseeded).

    Returns:
        Dict of noise mode -> evaluate_run() metrics.
    """
    return {mode: evaluate_run(agent, seed, n_episodes, mode, noise_std) for mode in modes}


def _run_dir(out_dir, name, seed):
    """Directory holding one run's outputs."""
    return Path(out_dir) / "runs" / f"{name}__seed{seed}"


def _stale(run_dir):
    """Whether a completed run was trained under another STUDY_VERSION."""
    with open(run_dir / "eval.json") as f:
        return json.load(f)["settings"].get("study_version") != STUDY_VERSION


def _needs_eval(run_dir, settings):
    """Whether a completed run lacks an evaluation matching the settings."""
    with open(run_dir / "eval.json") as f:
        evals = json.load(f)["eval"]
    for mode in NOISE_MODES:
        ev = evals.get(mode)
        std = 0.0 if mode == "none" else settings["noise_std"]
        if (not isinstance(ev, dict) or ev.get("episodes") != settings["eval_episodes"]
                or ev.get("noise_std") != std):
            return True
    return False


def _run_job(job):
    """Train (unless eval-only), evaluate and save one (config, seed) run.

    Module-level so ProcessPoolExecutor can pickle it.

    Args:
        job: Tuple of (config name, seed, settings dict, eval_only).  An
            eval-only job reloads policy.pth and replaces the 'eval' section of
            the existing eval.json.

    Returns:
        One-line completion message.
    """
    name, seed, settings, eval_only = job
    if settings["threads"]:
        torch.set_num_threads(settings["threads"])
    cfg = ALL_CONFIGS[name]
    run_dir = _run_dir(settings["out"], name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    if eval_only:
        with open(run_dir / "eval.json") as f:
            result = json.load(f)
        agent = load_agent(cfg, run_dir)
    else:
        agent, train_stats = train_run(cfg, seed, settings, run_dir)
        result = {
            "config": name,
            "label": cfg.label,
            "seed": seed,
            "components": {c: getattr(cfg, c) for c in COMPONENTS},
            "settings": {
                "episodes": settings["episodes"],
                "schedule_episodes": settings["schedule_episodes"],
                "device": str(device),
                "torch_threads": torch.get_num_threads(),
                "study_version": STUDY_VERSION,
            },
            "train": train_stats,
        }
    result["settings"]["noise_std"] = settings["noise_std"]
    result["eval"] = evaluate_all(agent, seed, settings["eval_episodes"], settings["noise_std"])
    # Written last: its presence marks the run as complete.
    with open(run_dir / "eval.json", "w") as f:
        json.dump(result, f, indent=2)
    evals = result["eval"]
    summary = ", ".join(f"{mode} {evals[mode]['harvested']:.2f} harvested/"
                        f"{evals[mode]['violation_steps']:.2f} violations" for mode in NOISE_MODES)
    action = "re-evaluated" if eval_only else f"trained in {result['train']['hours']:.2f} h"
    return f"[{name} seed {seed}] {action}: {summary} per episode"


def run_jobs(jobs, workers):
    """Execute jobs, reporting each failure without stopping the others.

    Args:
        jobs: List of (config name, seed, settings, eval_only) tuples.
        workers: Number of parallel processes; 1 runs in this process.
    """
    if workers <= 1:
        for job in jobs:
            try:
                print(_run_job(job), flush=True)
            except Exception:
                print(f"[{job[0]} seed {job[1]}] FAILED\n{traceback.format_exc()}", flush=True)
        return

    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            name, seed = futures[future][:2]
            try:
                print(future.result(), flush=True)
            except Exception:
                print(f"[{name} seed {seed}] FAILED\n{traceback.format_exc()}", flush=True)
