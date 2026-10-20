"""Figures of the ablation study."""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

# ─── Local ────────────────────────────────────────────────────────────────────
from experiments.ablation.config import (
    BASELINE, BASELINE_LABEL, NOISE_LABELS, NOISE_MODES, SEVERITY_TYPES, SEVERITY_UNITS,
    VIOLATION_LABELS, VIOLATION_NAMES, run_label
)

matplotlib.use('Agg')

# Light-surface chart tokens: one accent series against a gray reference.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_REF = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_ACCENT = "#2a78d6"
# Sequential blue ramp, steps 100-700, for magnitude heatmaps: the lightest
# step means "near zero" and recedes toward the surface.
_SEQ_BLUE = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
_SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE)
# Categorical slots 1-3, one per NOISE_MODES entry (validated for CVD separation).
_NOISE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def _style_axes(ax, grid_axis):
    """Apply recessive chrome: hairline axes, solid grid on one axis only."""
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_INK_2, labelsize=8.5, width=0.8, length=3)
    ax.grid(axis=grid_axis, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _plot_band(ax, x0, curves, color, label):
    """Plot the across-seed mean of ragged curves with a ±1 std wash."""
    stacked = np.full((len(curves), max(len(c) for c in curves)), np.nan)
    for i, c in enumerate(curves):
        stacked[i, :len(c)] = c
    mean = np.nanmean(stacked, axis=0)
    x = np.arange(x0, x0 + len(mean))
    ax.plot(x, mean, color=color, linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", label=label)
    if len(curves) > 1:
        std = np.nanstd(stacked, axis=0)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)


def plot_learning_curves(curves, out_path, window=100):
    """Plot each ablated variant against the full model as small multiples.

    One row per variant: harvested phycocyanin per episode (moving average) on
    the left and cumulative constraint-violation steps on the right, full
    model in gray.

    Args:
        curves: Output of summary._load_train_curves().
        out_path: PNG destination.
        window: Moving-average window for the harvest curves (episodes).
    """
    variants = [n for n in curves if n != "full"] or list(curves)
    has_ref = "full" in curves and variants != ["full"]
    window = min([window] + [len(s["harvested"]) for runs in curves.values() for s in runs])
    kernel = np.ones(window) / window

    def panels(name):
        runs = curves[name]
        return ([np.convolve(s["harvested"], kernel, mode="valid") for s in runs],
                [np.cumsum(s["violation_steps"]) for s in runs])

    fig, axes = plt.subplots(len(variants), 2, figsize=(11, 2.2 * len(variants) + 1.0),
                             sharex=True, sharey="col", squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    ref = panels("full") if has_ref else None
    for row, name in enumerate(variants):
        for col, (series, x0) in enumerate(zip(panels(name), (window, 1))):
            ax = axes[row, col]
            _style_axes(ax, "y")
            if ref is not None:
                _plot_band(ax, x0, ref[col], _REF, "Full model")
            _plot_band(ax, x0, series, _ACCENT, "Ablated variant")
        axes[row, 0].set_title(run_label(name), loc="left", fontsize=10,
                               fontweight="bold", color=_INK)
        axes[row, 0].set_ylabel(f"Harvested, g/ep ({window}-ep avg)", fontsize=9, color=_INK_2)
        axes[row, 1].set_ylabel("Cumulative violation steps", fontsize=9, color=_INK_2)
    for ax in axes[-1]:
        ax.set_xlabel("Training episode", fontsize=9, color=_INK_2)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    n_seeds = sorted({len(runs) for runs in curves.values()})
    seeds_note = "/".join(str(n) for n in n_seeds)
    fig.suptitle(f"Ablation learning curves (mean ± 1 std over {seeds_note} seeds)",
                 x=0.01, ha="left", fontsize=11, color=_INK)
    if has_ref:
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right",
                   ncol=2, frameon=False, fontsize=9, labelcolor=_INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _plot_grouped_panels(rows, panels, out_path, title):
    """Plot metrics as horizontal bars, one bar per noise mode in each config.

    Training metrics do not depend on the evaluation noise, so their panels
    show a single neutral bar per config.  Bars show the across-seed mean with
    a ±1 std whisker and a direct value label.

    Args:
        rows: Summary rows from summarize(), one per (config, noise mode).
        panels: Sequence of (summary key, panel title, value format).
        out_path: PNG destination.
        title: Figure title.
    """
    configs = list(dict.fromkeys(r["config"] for r in rows))
    by_key = {(r["config"], r["noise_mode"]): r for r in rows}
    n_modes = len(NOISE_MODES)
    bar_h = 0.8 / n_modes
    y = np.arange(len(configs))

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.6 * len(panels) + 1.5, 0.9 * len(configs) + 1.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(_SURFACE)
    for ax, (key, panel_title, fmt) in zip(axes, panels):
        _style_axes(ax, "x")
        if key.startswith("train_"):
            series = [(None, _REF, y)]
        else:
            series = [(mode, color, y + (i - (n_modes - 1) / 2) * bar_h)
                      for i, (mode, color) in enumerate(zip(NOISE_MODES, _NOISE_COLORS))]

        bars = []
        for mode, color, yy in series:
            src = [by_key.get((c, mode or NOISE_MODES[0]), {}) for c in configs]
            means = np.array([np.nan if r.get(f"{key}_mean") is None else r[f"{key}_mean"]
                              for r in src])
            stds = np.array([0.0 if r.get(f"{key}_std") is None else r[f"{key}_std"]
                             for r in src])
            # 2px surface gap between adjacent bars.
            ax.barh(yy, np.nan_to_num(means), height=bar_h, color=color,
                    edgecolor=_SURFACE, linewidth=1.0,
                    label=NOISE_LABELS[mode] if mode else None)
            spread = stds > 0  # single-seed configs have no error bar
            ax.errorbar(means[spread], yy[spread], xerr=stds[spread], fmt="none",
                        ecolor=_INK_2, elinewidth=0.8, capsize=2)
            bars.append((yy, means, stds))

        tops = [m + s for _, means, stds in bars for m, s in zip(means, stds) if np.isfinite(m)]
        right = max(tops) if tops and max(tops) > 0 else 1.0
        for yy, means, stds in bars:
            for yi, m, s in zip(yy, means, stds):
                if np.isfinite(m):
                    ax.text(m + s + 0.03 * right, yi, fmt.format(m), va="center",
                            fontsize=7.5, color=_INK_2)
        ax.set_xlim(0, right * 1.35)
        ax.set_title(panel_title, loc="left", fontsize=10, color=_INK)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if right >= 1000:
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([run_label(c) for c in configs], fontsize=9, color=_INK)
    axes[0].invert_yaxis()

    # Noise-mode legend, unless every panel is a training metric.
    eval_axes = [ax for ax, (k, _, _) in zip(axes, panels) if not k.startswith("train_")]
    if eval_axes:
        handles, labels = eval_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False,
                   fontsize=9, labelcolor=_INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def plot_final_metrics(rows, out_path):
    """Plot the final evaluation metrics of every config under each noise mode.

    Args:
        rows: Summary rows from summarize(), one per (config, noise mode).
        out_path: PNG destination.
    """
    _plot_grouped_panels(rows, (
        ("harvested", "Phycocyanin harvested (g/ep)", "{:.2f}"),
        ("violation_steps", "Violations per episode", "{:.2f}"),
        ("violation_episode_pct", "Episodes with a violation (%)", "{:.1f}"),
        ("apn_intervention_rate", "Steps corrected by the APN", "{:.2f}"),
        ("train_hours", "Training time (h)", "{:.2f}"),
    ), out_path, "Ablation study (evaluation with multipliers frozen at 0)")


def plot_learning_speed(rows, out_path, target, levels, window, early):
    """Plot how fast every config learns, from its training curves.

    Args:
        rows: Summary rows from summarize(), one per (config, noise mode).
        out_path: PNG destination.
        target: The full model's final harvest (g/ep) the levels are shares of.
        levels: Fractions of target behind the train_eps_to_<pct> metrics.
        window: Moving-average window of the episodes-to-level metrics.
        early: Episodes in the early-harvest metric.
    """
    _plot_grouped_panels(rows, [
        (f"train_eps_to_{round(100 * lvl)}",
         f"Episodes to {round(100 * lvl)}% ({lvl * target:.2f} g/ep)", "{:,.0f}")
        for lvl in levels] + [
        ("train_auc_early", f"Mean harvest, first {early:,} eps (g/ep)", "{:.2f}"),
        ("train_auc", "Mean harvest, whole run (g/ep)", "{:.2f}"),
    ], out_path, f"Learning speed ({window}-episode average harvest vs Full's final "
                 f"{target:.2f} g/ep; lower episodes and higher harvest are better)")


def _compact(v):
    """Format a heatmap cell value in at most ~5 characters."""
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _plot_heatmaps(rows, panels, out_path, title, n_cols=2):
    """Plot metrics as config x noise-mode heatmaps, one panel per metric.

    Each panel has its own sequential scale from 0 (metrics differ in
    magnitude) with a colorbar, and every cell carries its value in ink; a
    cell without data shows '-'.

    Args:
        rows: Summary rows from summarize(), one per (config, noise mode).
        panels: Sequence of (summary key, panel title, cell formatter).
        out_path: PNG destination.
        title: Figure title.
        n_cols: Panels per figure row.
    """
    configs = list(dict.fromkeys(r["config"] for r in rows))
    present = {r["noise_mode"] for r in rows}
    modes = [m for m in NOISE_MODES if m in present]
    by_key = {(r["config"], r["noise_mode"]): r for r in rows}

    n_cols = min(n_cols, len(panels))
    n_rows = int(np.ceil(len(panels) / n_cols))
    panel_w = 0.9 * len(modes) + 2.8
    panel_h = 0.36 * len(configs) + 1.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows + 0.6),
                             squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    for ax, (key, panel_title, fmt) in zip(axes.flat, panels):
        vals = np.array([[np.nan if by_key.get((c, m), {}).get(f"{key}_mean") is None
                          else by_key[(c, m)][f"{key}_mean"] for m in modes]
                         for c in configs], dtype=float)
        finite = vals[np.isfinite(vals)]
        top = float(finite.max()) if finite.size else 0.0
        norm = Normalize(vmin=0.0, vmax=top if top > 0 else 1.0)
        ax.set_facecolor(_SURFACE)
        mesh = ax.pcolormesh(np.ma.masked_invalid(vals), cmap=_SEQ_CMAP, norm=norm,
                             edgecolors=_SURFACE, linewidth=2)
        for i in range(len(configs)):
            for j in range(len(modes)):
                v = vals[i, j]
                dark = np.isfinite(v) and norm(v) > 0.55
                ax.text(j + 0.5, i + 0.5, "-" if not np.isfinite(v) else fmt(v),
                        ha="center", va="center", fontsize=7,
                        color=_SURFACE if dark else _INK)
        ax.set_xticks(np.arange(len(modes)) + 0.5)
        ax.set_xticklabels([NOISE_LABELS[m] for m in modes], fontsize=8, color=_INK_2)
        ax.set_yticks(np.arange(len(configs)) + 0.5)
        ax.set_yticklabels([run_label(c) for c in configs], fontsize=8.5, color=_INK)
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(panel_title, loc="left", fontsize=10, color=_INK)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.015)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=7.5, colors=_INK_2, length=0)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def plot_occurrence(rows, out_path):
    """Plot how often each constraint is violated, per config and noise mode.

    Each value is the mean number of violation events per evaluation episode,
    as the environment counts them, averaged over seeds.

    Args:
        rows: Summary rows from summarize(), one per (config, noise mode).
        out_path: PNG destination.
    """
    _plot_heatmaps(rows, [("violation_steps", "All constraints", _compact)]
                   + [(f"{n}_events", VIOLATION_LABELS[n], _compact)
                      for n in VIOLATION_NAMES.values()],
                   out_path, "Constraint occurrence: violation events per episode "
                             "(mean over seeds)")


def plot_severity(rows, out_path):
    """Plot how far each constraint is violated, per config and noise mode.

    One row of panels per constraint with a magnitude: on the left the worst
    exceedance over all evaluation episodes of a run, on the right the mean
    exceedance per violation event, both as a percentage of the violated
    limit and averaged over seeds ('-' where the constraint never fired).

    Args:
        rows: Summary rows from summarize(), one per (config, noise mode).
        out_path: PNG destination.
    """
    panels = []
    for n in SEVERITY_TYPES:
        panels.append((f"{n}_peak_pct", f"{VIOLATION_LABELS[n]}: worst ({SEVERITY_UNITS[n]})",
                       _compact))
        panels.append((f"{n}_mean_depth_pct",
                       f"{VIOLATION_LABELS[n]}: mean per event ({SEVERITY_UNITS[n]})", _compact))
    _plot_heatmaps(rows, panels, out_path,
                   "Constraint severity: exceedance as % of the limit (mean over seeds)")


def plot_safe_vs_baseline(rows, out_path):
    """Compare the full Safe RL agent with the Standard RL baseline under every noise mode.

    One figure, one panel per KPI: grouped bars per noise mode, Safe RL in the
    accent color and the baseline in gray, across-seed mean with a ±1 std
    whisker and a direct value label.

    Args:
        rows: Summary rows from summarize(), one per (config, noise mode).
        out_path: PNG destination.

    Returns:
        False (nothing written) unless both the full model and the baseline
        have results.
    """
    by_key = {(r["config"], r["noise_mode"]): r for r in rows}
    modes = [m for m in NOISE_MODES if ("full", m) in by_key and (BASELINE, m) in by_key]
    if not modes:
        return False
    agents = (("full", "Safe RL", _ACCENT), (BASELINE, BASELINE_LABEL, _REF))
    panels = (("harvested", "Phycocyanin harvested (g/ep)", "{:.2f}"),
              ("violation_steps", "Violations per episode", "{:.2f}"),
              ("violation_episode_pct", "Episodes with a violation (%)", "{:.1f}"))

    x = np.arange(len(modes))
    bar_w = 0.38
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(_SURFACE)
    for ax, (key, title, fmt) in zip(axes, panels):
        _style_axes(ax, "y")
        bars = []
        for i, (name, label, color) in enumerate(agents):
            xx = x + (i - 0.5) * bar_w
            means = np.array([by_key[(name, m)][f"{key}_mean"] or 0.0 for m in modes])
            stds = np.array([by_key[(name, m)][f"{key}_std"] or 0.0 for m in modes])
            ax.bar(xx, means, width=bar_w, color=color, edgecolor=_SURFACE, linewidth=1.0,
                   label=label)
            spread = stds > 0  # single-seed runs have no error bar
            ax.errorbar(xx[spread], means[spread], yerr=stds[spread], fmt="none",
                        ecolor=_INK_2, elinewidth=0.8, capsize=2)
            bars.append((xx, means, stds))
        top = max([m + s for _, ms, ss in bars for m, s in zip(ms, ss)] + [0.0]) or 1.0
        for xx, means, stds in bars:
            for xi, m, s in zip(xx, means, stds):
                ax.text(xi, m + s + 0.02 * top, fmt.format(m), ha="center", va="bottom",
                        fontsize=7.5, color=_INK_2)
        ax.set_ylim(0, top * 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels([NOISE_LABELS[m] for m in modes], fontsize=9, color=_INK)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(title, loc="left", fontsize=10, color=_INK)

    n_seeds = sorted({by_key[(n, m)]["seeds"] for n, _, _ in agents for m in modes})
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2,
               frameon=False, fontsize=9, labelcolor=_INK_2)
    fig.suptitle("Safe RL vs Standard RL baseline (mean ± 1 std over "
                 f"{'/'.join(str(n) for n in n_seeds)} seeds; multipliers frozen at 0)",
                 x=0.01, ha="left", fontsize=11, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return True
