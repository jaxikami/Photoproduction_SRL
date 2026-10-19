"""Command line of the ablation study (python ablation.py --help)."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import argparse
import os
import warnings

# ─── Local ────────────────────────────────────────────────────────────────────
from agents.networks import device
from experiments.ablation import summary
from experiments.ablation.config import (
    ALL_CONFIGS, DEFAULT_EPISODES, DEFAULT_EVAL_EPISODES, DEFAULT_NOISE_STD, DEFAULT_SEEDS,
    DEFAULT_WORKERS, LOO_CONFIGS, NOISE_MODES, SCHEDULE_EPISODES
)
from experiments.ablation.runner import _needs_eval, _run_dir, _stale, run_jobs

warnings.filterwarnings("ignore", category=UserWarning,
                        message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*")


def _parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Ablation study of the Safe RL agent "
                    "(GRU memory, Lagrangian, stage mask, APN projection, safety gradient).",
        epilog="configs: " + ", ".join(ALL_CONFIGS),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--design", choices=("loo", "factorial"), default="loo",
                        help="loo: full model plus one run per removed component; "
                             f"factorial: all {len(ALL_CONFIGS)} combinations")
    parser.add_argument("--configs", nargs="+", choices=list(ALL_CONFIGS), metavar="NAME",
                        help="run only these configs instead of the --design set")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="each config is trained once per seed")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="training episodes per run (fixed budget, no early stopping)")
    parser.add_argument("--schedule-episodes", type=int, default=SCHEDULE_EPISODES,
                        help="horizon of the LR/entropy schedules (main.py's MAX_EPISODES)")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES,
                        help="evaluation episodes per trained policy and noise mode")
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD,
                        help="relative std of the action and state noise used in evaluation")
    parser.add_argument("--out", default="ablation", help="output directory")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="parallel training processes (they share the GPU); 1 shows "
                             "progress bars")
    parser.add_argument("--force", action="store_true",
                        help="re-run configs that already have results")
    parser.add_argument("--summarize-only", action="store_true",
                        help="skip training; aggregate existing runs and re-plot")
    return parser.parse_args()


def main():
    """Run the requested ablation runs, then summarize everything on disk."""
    args = _parse_args()
    if not args.summarize_only:
        names = args.configs or (LOO_CONFIGS if args.design == "loo" else list(ALL_CONFIGS))
        settings = {
            "out": args.out,
            "episodes": args.episodes,
            "schedule_episodes": args.schedule_episodes,
            "eval_episodes": args.eval_episodes,
            "noise_std": args.noise_std,
            "progress": args.workers <= 1,
            # Split the cores between workers so they don't oversubscribe.
            "threads": (max(1, (os.cpu_count() or 1) // args.workers)
                        if args.workers > 1 else None),
        }
        jobs = [(name, seed) for name in names for seed in args.seeds]
        to_train, to_eval = [], []
        for name, seed in jobs:
            run_dir = _run_dir(args.out, name, seed)
            if args.force or not (run_dir / "eval.json").exists() or _stale(run_dir):
                to_train.append((name, seed))
            elif _needs_eval(run_dir, settings):
                to_eval.append((name, seed))

        print(f"--- Ablation: {len(names)} configs x {len(args.seeds)} seeds = {len(jobs)} runs, "
              f"{len(to_train)} to train, {len(to_eval)} to re-evaluate "
              f"({len(jobs) - len(to_train) - len(to_eval)} already complete) ---")
        print(f"  configs: {', '.join(names)}")
        print(f"  {args.episodes:,} training episodes per run "
              f"(schedule horizon {args.schedule_episodes:,}), "
              f"{args.eval_episodes} evaluation episodes per noise mode: "
              f"{', '.join(NOISE_MODES)} (std {args.noise_std:g})")
        print(f"  device: {device} | workers: {args.workers}")

        run_jobs([(name, seed, settings, False) for name, seed in to_train]
                 + [(name, seed, settings, True) for name, seed in to_eval], args.workers)
    summary.summarize(args.out)


if __name__ == "__main__":
    main()
