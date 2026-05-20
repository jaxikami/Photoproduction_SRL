"""
Utility module for metrics tracking and visualization.

Provides DataLogger for recording episode trajectories, rewards, and
constraint violations during both training and evaluation phases.
Includes the Plotter class for generating comparative visualizations
of agent performance and safety integrity.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

# Configure matplotlib for research paper quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2
})


class DataLogger:
    """
    Centralized logging architecture for recording agent metrics.
    Tracks g1–g4 constraint violations plus idle-stage compliance.
    """

    def __init__(self):
        self.training_log = {"Standard RL": [], "SPRL": []}
        self.training_violations = {"Standard RL": [], "SPRL": []}

        self.eval_data = {"Standard RL": None, "SPRL": None}
        self.eval_violations = {"Standard RL": [], "SPRL": []}
        self.eval_violations_details = {
            "Standard RL": {"g1": [], "g2": [], "g3": [], "g4": []},
            "SPRL":        {"g1": [], "g2": [], "g3": [], "g4": []}
        }

    def log_training_episode(self, agent_name, total_reward, violation_count=0):
        self.training_log[agent_name].append(total_reward)
        self.training_violations[agent_name].append(violation_count)

    def log_evaluation_trajectory(self, agent_name, states, actions, rewards, info_list, agg_data=None):
        self.eval_data[agent_name] = {
            "states":  np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "is_safe": np.array([1 if i["violation_count"] == 0 else 0 for i in info_list]),
            "metrics": pd.DataFrame(info_list),
            "agg_data": agg_data
        }

    def log_evaluation_episode_violations(self, agent_name, violation_count,
                                           g1_count=0, g2_count=0, g3_count=0,
                                           g4_count=0):
        self.eval_violations[agent_name].append(violation_count)
        self.eval_violations_details[agent_name]["g1"].append(g1_count)
        self.eval_violations_details[agent_name]["g2"].append(g2_count)
        self.eval_violations_details[agent_name]["g3"].append(g3_count)
        self.eval_violations_details[agent_name]["g4"].append(g4_count)


class Plotter:
    """Static utility class for generating matplotlib visualizations."""

    @staticmethod
    def plot_training_results(training_log, agent_name, window=50):
        rewards = training_log.get(agent_name, [])
        if len(rewards) < window:
            return

        plt.figure(figsize=(10, 6))
        mv_avg = pd.Series(rewards).rolling(window=window).mean()
        plt.plot(mv_avg, label=f"{agent_name} (MA {window})", color='tab:blue')
        plt.plot(rewards, alpha=0.15, color='tab:blue')

        valid_mv_avg = mv_avg.dropna()
        if len(valid_mv_avg) > 0:
            y_max = valid_mv_avg.max()
            y_min = -2.0
            y_range = y_max - y_min
            plt.ylim(y_min, y_max + 0.1 * y_range)

        plt.title(f"Training Convergence: {agent_name}")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs("plot", exist_ok=True)
        plt.savefig(os.path.join("plot", f"training_{agent_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_training_violations(logger):
        agents = ["Standard RL", "SPRL"]
        train_viols = [sum(logger.training_violations.get(a, [])) for a in agents]

        fig, ax1 = plt.subplots(figsize=(6, 5))
        ax1.bar(agents, train_viols, color=['tab:blue', 'tab:orange'])
        ax1.set_title("Total Training Violations")
        ax1.set_xlabel("Agent")
        ax1.set_ylabel("Number of Violations")
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        os.makedirs("plot", exist_ok=True)
        plt.savefig(os.path.join("plot", "training_violations.png"), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_comprehensive_evaluation(eval_data, eval_violations):
        """
        Generates a 10-panel subplot comparing Standard RL and SPRL agents:
        Row 1: Nitrate levels (Standard RL / SPRL)
        Row 2: Violations bar / Average Production
        Row 3: Light Intensity / Nitrate Feed
        Row 4: Volume trajectories (Standard RL / SPRL)
        Row 5: cq/cx Ratio (Standard RL / SPRL)
        """
        if "Standard RL" not in eval_data or "SPRL" not in eval_data:
            print("Evaluation data missing. Cannot plot.")
            return

        nr_data   = eval_data["Standard RL"]
        sprl_data = eval_data["SPRL"]

        # Constants
        I_MIN, I_MAX = 120.0, 400.0
        FN_MAX_GROWTH = 40.0
        FN_MAX_PROD   = 10.0
        N_LIMIT_PATH  = 800.0
        N_LIMIT_TERM  = 150.0
        RATIO_LIMIT   = 0.011
        V_MAX         = 50.0
        V_MIN         = 5.0

        fig, axes = plt.subplots(5, 2, figsize=(15, 22))

        if nr_data is not None and len(nr_data["states"]) > 0:
            time = np.arange(len(nr_data["states"]))
        elif sprl_data is not None and len(sprl_data["states"]) > 0:
            time = np.arange(len(sprl_data["states"]))
        else:
            return

        # ----- Row 1: Nitrate Levels -----
        for col, (name, data, color) in enumerate([
            ("Standard RL", nr_data, 'tab:blue'),
            ("SPRL", sprl_data, 'tab:orange')
        ]):
            ax = axes[0, col]
            if data is not None:
                n_best = data["states"][:, 1] * 800.0
                ax.plot(time, n_best, label="Best Run", color=color)
                agg = data.get("agg_data", {})
                if agg:
                    ax.fill_between(time, agg["nitrate_min"] * 800.0,
                                    agg["nitrate_max"] * 800.0,
                                    color=color, alpha=0.2, label="Min/Max")
            ax.axhline(y=N_LIMIT_PATH, color='r', linestyle='--', alpha=0.5, label="$g_1$")
            ax.axhline(y=N_LIMIT_TERM, color='darkred', linestyle='--', alpha=0.5, label="$g_3$")
            ax.set_title(f"Nitrate ($c_N$) — {name}")
            ax.set_ylabel("mg/L")
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=9)

        # ----- Row 2: Violations / Production -----
        ax = axes[1, 0]
        agents = ["Standard RL", "SPRL"]
        viols = [int(np.sum(eval_violations.get(a, []))) for a in agents]
        bars = ax.bar(agents, viols, color=['tab:blue', 'tab:orange'])
        ax.set_title("Total Evaluation Violations")
        ax.set_ylabel("Count")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval),
                    ha='center', va='bottom')

        ax = axes[1, 1]
        if nr_data is not None and nr_data.get("agg_data"):
            ax.plot(time, nr_data["agg_data"]["production_avg"] * 0.2,
                    label="Standard RL", color='tab:blue')
        if sprl_data is not None and sprl_data.get("agg_data"):
            ax.plot(time, sprl_data["agg_data"]["production_avg"] * 0.2,
                    label="SPRL", color='tab:orange')
        ax.set_title("Average Phycocyanin ($c_q$)")
        ax.set_ylabel("g/L")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.2)
        ax.legend()

        # ----- Row 3: Actions -----
        ax = axes[2, 0]
        if nr_data is not None:
            nr_I = I_MIN + ((nr_data["actions"][:, 1] + 1.0) / 2.0) * (I_MAX - I_MIN)
            t_act = np.arange(len(nr_I) + 1)
            ax.step(t_act, np.append(nr_I, nr_I[-1]), where='post',
                    label="Standard RL", color='tab:blue')
        if sprl_data is not None:
            s_I = I_MIN + ((sprl_data["actions"][:, 1] + 1.0) / 2.0) * (I_MAX - I_MIN)
            t_act = np.arange(len(s_I) + 1)
            ax.step(t_act, np.append(s_I, s_I[-1]), where='post',
                    label="SPRL", color='tab:orange')
        ax.set_title("Light Intensity ($I$)")
        ax.set_ylabel(r"$\mu mol/m^2/s$")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.2)
        ax.legend()

        ax = axes[2, 1]
        if nr_data is not None:
            nr_Fn = ((nr_data["actions"][:, 2] + 1.0) / 2.0) * FN_MAX_GROWTH
            t_act = np.arange(len(nr_Fn) + 1)
            ax.step(t_act, np.append(nr_Fn, nr_Fn[-1]), where='post',
                    label="Standard RL", color='tab:blue')
        if sprl_data is not None:
            s_Fn = ((sprl_data["actions"][:, 2] + 1.0) / 2.0) * FN_MAX_GROWTH
            t_act = np.arange(len(s_Fn) + 1)
            ax.step(t_act, np.append(s_Fn, s_Fn[-1]), where='post',
                    label="SPRL", color='tab:orange')
        ax.set_title("Nitrate Feed ($F_N$)")
        ax.set_ylabel("mg/L/h")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.2)
        ax.legend()

        # ----- Row 4: Volume Trajectories -----
        for col, (name, data, color) in enumerate([
            ("Standard RL", nr_data, 'tab:blue'),
            ("SPRL", sprl_data, 'tab:orange')
        ]):
            ax = axes[3, col]
            if data is not None:
                v_best = data["states"][:, 3] * V_MAX
                ax.plot(time, v_best, label="Best Run", color=color)
                agg = data.get("agg_data", {})
                if agg and "volume_min" in agg:
                    ax.fill_between(time, agg["volume_min"] * V_MAX,
                                    agg["volume_max"] * V_MAX,
                                    color=color, alpha=0.2, label="Min/Max")
            ax.axhline(y=V_MAX, color='r', linestyle='--', alpha=0.5, label="$g_4$")
            ax.axhline(y=V_MIN, color='darkred', linestyle='--', alpha=0.5, label="$g_5$")
            ax.set_title(f"Reactor Volume — {name}")
            ax.set_ylabel("L")
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=9)

        # ----- Row 5: Ratio -----
        for col, (name, data, color) in enumerate([
            ("Standard RL", nr_data, 'tab:blue'),
            ("SPRL", sprl_data, 'tab:orange')
        ]):
            ax = axes[4, col]
            if data is not None and data.get("agg_data"):
                agg = data["agg_data"]
                ax.fill_between(time, agg["ratio_min"], agg["ratio_max"],
                                color=color, alpha=0.2, label="Min/Max")
                ax.plot(time, agg["ratio_avg"], color=color, label="Mean")
            ax.axhline(y=RATIO_LIMIT, color='r', linestyle='--', alpha=0.5, label="$g_2$")
            ax.set_title(f"$c_q / c_x$ Ratio — {name}")
            ax.set_ylabel("Ratio")
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=9)

        plt.tight_layout()
        os.makedirs("plot", exist_ok=True)
        plt.savefig(os.path.join("plot", "comprehensive_evaluation.png"), dpi=300, bbox_inches='tight')
        plt.close()
