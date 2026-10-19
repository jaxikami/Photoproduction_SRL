"""Aggregation of completed runs: summary.csv, console tables and the plots."""

# ─── Standard Library ─────────────────────────────────────────────────────────
import csv
import json
from pathlib import Path

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from experiments.ablation.config import (
    BASELINE, NOISE_LABELS, NOISE_MODES, RUN_NAMES, SEVERITY_TYPES, VIOLATION_NAMES, run_label
)
from experiments.ablation.plots import (
    plot_final_metrics, plot_learning_curves, plot_occurrence, plot_safe_vs_baseline,
    plot_severity
)
from experiments.ablation.runner import _run_dir

# Evaluation metrics aggregated across seeds, once per noise mode.
EVAL_METRICS = (["harvested", "violation_steps", "violation_episode_pct", "ended_idle_pct",
                 "return", "illegal_cmd_rate", "apn_intervention_rate", "apn_avg_iters"]
                + [f"{n}_events" for n in VIOLATION_NAMES.values()]
                + [f"{n}_episode_pct" for n in VIOLATION_NAMES.values()]
                + ["peak_violation_pct"] + [f"{n}_peak_pct" for n in SEVERITY_TYPES]
                + [f"{n}_mean_depth_pct" for n in SEVERITY_TYPES])
# Training metrics aggregated across seeds: (summary prefix, key in 'train').
TRAIN_METRICS = [("train_violation_steps", "violation_steps_total"),
                 ("train_hours", "hours"),
                 ("s_per_episode", "s_per_episode")]
SUMMARY_KEYS = EVAL_METRICS + [key for key, _ in TRAIN_METRICS]


def load_results(out_dir):
    """Load every completed run (variants, then the baseline), grouped by config.

    Args:
        out_dir: Study output directory.

    Returns:
        Dict of config name -> list of eval.json dicts sorted by seed.
    """
    grouped = {}
    for path in Path(out_dir, "runs").glob("*/eval.json"):
        with open(path) as f:
            result = json.load(f)
        grouped.setdefault(result["config"], []).append(result)
    return {name: sorted(grouped[name], key=lambda r: r["seed"])
            for name in RUN_NAMES if name in grouped}


def _load_train_curves(out_dir, results):
    """Read the per-episode harvest and violation columns of each run.

    Returns:
        Dict of config name -> list (one per seed) of dicts with 'harvested'
        and 'violation_steps' arrays.
    """
    curves = {}
    for name, runs in results.items():
        for run in runs:
            with open(_run_dir(out_dir, name, run["seed"]) / "train_log.csv", newline="") as f:
                records = list(csv.DictReader(f))
            curves.setdefault(name, []).append({
                "harvested": np.array([float(r["harvested"]) for r in records]),
                "violation_steps": np.array([float(r["violation_steps"]) for r in records]),
            })
    return curves


def _fmt(mean, std, spec):
    """Format 'mean ± std' for the console table; '-' when unavailable."""
    if mean is None:
        return "-"
    if std is None:
        return format(mean, spec)
    return f"{format(mean, spec)} ± {format(std, spec)}"


def summarize(out_dir):
    """Aggregate completed runs across seeds, then write the table and plots.

    Args:
        out_dir: Study output directory.
    """
    results = load_results(out_dir)
    if not results:
        print(f"No completed runs under {Path(out_dir, 'runs')}.")
        return

    def aggregate(vals):
        vals = [v for v in vals if v is not None]
        return (float(np.mean(vals)) if vals else None,
                float(np.std(vals, ddof=1)) if len(vals) > 1 else None)

    rows = []
    for name, runs in results.items():
        for mode in NOISE_MODES:
            evals = [r["eval"][mode] for r in runs if isinstance(r["eval"].get(mode), dict)]
            if not evals:
                continue
            row = {"config": name, "label": run_label(name),
                   "noise_mode": mode, "seeds": len(evals)}
            for key in EVAL_METRICS:
                row[f"{key}_mean"], row[f"{key}_std"] = aggregate([e.get(key) for e in evals])
            for key, field in TRAIN_METRICS:
                row[f"{key}_mean"], row[f"{key}_std"] = aggregate(
                    [r["train"].get(field) for r in runs])
            rows.append(row)

    full = {r["noise_mode"]: r for r in rows if r["config"] == "full"}
    for row in rows:
        ref = full.get(row["noise_mode"])
        row["harvested_vs_full_pct"] = (
            100.0 * (row["harvested_mean"] / ref["harvested_mean"] - 1.0)
            if ref and ref["harvested_mean"] and row["harvested_mean"] is not None else None)

    fields = (["config", "label", "noise_mode", "seeds", "harvested_vs_full_pct"]
              + [f"{key}_{stat}" for key in SUMMARY_KEYS for stat in ("mean", "std")])
    with open(Path(out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    rule = "=" * 120
    short = {"path_nitrate": "G1", "product_ratio": "G2", "terminal_nitrate": "G3",
             "overflow": "G4", "end_idle": "G5"}
    names = list(VIOLATION_NAMES.values())
    for mode in NOISE_MODES:
        mode_rows = [r for r in rows if r["noise_mode"] == mode]
        if not mode_rows:
            continue
        print(f"\n{rule}")
        print(f"  ABLATION SUMMARY ({NOISE_LABELS[mode].lower()}): evaluation with "
              f"multipliers frozen at 0 (mean ± std over seeds)")
        print(rule)
        print(f"  {'Config':<26} {'Seeds':>5} {'Harvested (g/ep)':>18} {'vs Full':>8} "
              f"{'Violations/ep':>15} {'Viol eps (%)':>13} {'Ends idle (%)':>14} "
              f"{'Worst exceed (%)':>18}")
        print(f"  {'-' * 26} {'-' * 5} {'-' * 18} {'-' * 8} {'-' * 15} {'-' * 13} {'-' * 14} "
              f"{'-' * 18}")
        for row in mode_rows:
            delta = ("-" if row["harvested_vs_full_pct"] is None
                     else f"{row['harvested_vs_full_pct']:+.1f}%")
            print(f"  {row['label']:<26} {row['seeds']:>5} "
                  f"{_fmt(row['harvested_mean'], row['harvested_std'], '.2f'):>18} {delta:>8} "
                  f"{_fmt(row['violation_steps_mean'], row['violation_steps_std'], '.2f'):>15} "
                  f"{_fmt(row['violation_episode_pct_mean'], row['violation_episode_pct_std'], '.1f'):>13} "
                  f"{_fmt(row['ended_idle_pct_mean'], row['ended_idle_pct_std'], '.1f'):>14} "
                  f"{_fmt(row['peak_violation_pct_mean'], row['peak_violation_pct_std'], '.2f'):>18}")
        # Occurrence and severity per constraint (means over seeds).
        print("\n  Violation events per episode by constraint")
        print(f"  {'Config':<26} " + " ".join(f"{short[n]:>9}" for n in names))
        for row in mode_rows:
            print(f"  {row['label']:<26} " + " ".join(
                f"{_fmt(row[f'{n}_events_mean'], None, '.2f'):>9}" for n in names))
        print("\n  Worst exceedance by constraint (% of limit; mean per event in brackets)")
        print(f"  {'Config':<26} " + " ".join(f"{short[n]:>15}" for n in SEVERITY_TYPES))
        for row in mode_rows:
            cells = []
            for n in SEVERITY_TYPES:
                peak, avg = row[f"{n}_peak_pct_mean"], row[f"{n}_mean_depth_pct_mean"]
                cells.append("-" if not peak else
                             f"{peak:.1f} ({'-' if avg is None else format(avg, '.1f')})")
            print(f"  {row['label']:<26} " + " ".join(f"{c:>15}" for c in cells))
        for row in mode_rows:
            if row["illegal_cmd_rate_mean"]:
                print(f"  {row['label']}: the interlock blocked stage-illegal commands on "
                      f"{100 * row['illegal_cmd_rate_mean']:.1f}% of locked action dims")
    _print_safe_vs_baseline(rows, rule)
    print(f"{rule}\n")

    plot_learning_curves(_load_train_curves(out_dir, results), Path(out_dir, "ablation_curves.png"))
    plot_final_metrics(rows, Path(out_dir, "ablation_final.png"))
    plot_occurrence(rows, Path(out_dir, "ablation_occurrence.png"))
    plot_severity(rows, Path(out_dir, "ablation_severity.png"))
    written = ["curves", "final", "occurrence", "severity"]
    if plot_safe_vs_baseline(rows, Path(out_dir, "ablation_vs_baseline.png")):
        written.append("vs_baseline")
    print(f"Wrote {Path(out_dir, 'summary.csv')} and " + ", ".join(
        str(Path(out_dir, f"ablation_{n}.png")) for n in written))



def _print_safe_vs_baseline(rows, rule):
    """Console table: the full Safe RL agent against the baseline per noise mode."""
    by_key = {(r["config"], r["noise_mode"]): r for r in rows}
    modes = [m for m in NOISE_MODES if ("full", m) in by_key and (BASELINE, m) in by_key]
    if not modes:
        return
    print(f"\n{rule}")
    print("  SAFE RL vs STANDARD RL BASELINE (mean ± std over seeds)")
    print(rule)
    print(f"  {'Noise':<14} {'Agent':<14} {'Harvested (g/ep)':>18} {'Violations/ep':>15} "
          f"{'Viol eps (%)':>13} {'Ends idle (%)':>14} {'Worst exceed (%)':>18}")
    for mode in modes:
        for name in ("full", BASELINE):
            r = by_key[(name, mode)]
            agent = "Safe RL" if name == "full" else run_label(name)
            print(f"  {NOISE_LABELS[mode]:<14} {agent:<14} "
                  f"{_fmt(r['harvested_mean'], r['harvested_std'], '.2f'):>18} "
                  f"{_fmt(r['violation_steps_mean'], r['violation_steps_std'], '.2f'):>15} "
                  f"{_fmt(r['violation_episode_pct_mean'], r['violation_episode_pct_std'], '.1f'):>13} "
                  f"{_fmt(r['ended_idle_pct_mean'], r['ended_idle_pct_std'], '.1f'):>14} "
                  f"{_fmt(r['peak_violation_pct_mean'], r['peak_violation_pct_std'], '.2f'):>18}")
