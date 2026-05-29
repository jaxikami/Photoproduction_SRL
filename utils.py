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
    """Centralized logging architecture for recording agent metrics.
    
    Tracks rewards, trajectories, g1-g4 constraint violations, and idle-stage
    compliance (g5) during both training and evaluation.
    """

    def __init__(self):
        """Initializes empty dictionaries to hold tracking metrics."""
        self.training_log = {"Standard RL": [], "safe_RL agent": []}
        self.training_violations = {"Standard RL": [], "safe_RL agent": []}

        self.eval_data = {"Standard RL": None, "safe_RL agent": None}
        self.eval_violations = {"Standard RL": [], "safe_RL agent": []}
        self.eval_violations_details = {
            "Standard RL": {"g1": [], "g2": [], "g3": [], "g4": [], "g5": []},
            "safe_RL agent":        {"g1": [], "g2": [], "g3": [], "g4": [], "g5": []}
        }

    def log_training_episode(self, agent_name, total_reward, violation_count=0):
        """Logs the final total reward and violation count for a training episode.

        Args:
            agent_name (str): Name of the agent.
            total_reward (float): Total reward achieved during the episode.
            violation_count (int, optional): Number of constraint violations. Defaults to 0.
        """
        self.training_log[agent_name].append(total_reward)
        self.training_violations[agent_name].append(violation_count)

    def log_evaluation_trajectory(self, agent_name, states, actions, rewards, info_list, agg_data=None):
        """Stores the detailed rollout trajectory from an evaluation run.

        Args:
            agent_name (str): Name of the evaluated agent.
            states (list): Sequence of state observations.
            actions (list): Sequence of executed actions.
            rewards (list): Sequence of rewards received.
            info_list (list): Sequence of info dictionaries.
            agg_data (dict, optional): Aggregated min/max/avg statistical data. Defaults to None.
        """
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
                                           g4_count=0, g5_count=0):
        """Records granular constraint-specific violations for an evaluation episode.

        Args:
            agent_name (str): Name of the evaluated agent.
            violation_count (int): Total number of violations.
            g1_count (int, optional): Nitrate path violations. Defaults to 0.
            g2_count (int, optional): Product/biomass ratio violations. Defaults to 0.
            g3_count (int, optional): Terminal nitrate violations. Defaults to 0.
            g4_count (int, optional): Reactor overflow violations. Defaults to 0.
            g5_count (int, optional): Idle stage bounds violations. Defaults to 0.
        """
        self.eval_violations[agent_name].append(violation_count)
        self.eval_violations_details[agent_name]["g1"].append(g1_count)
        self.eval_violations_details[agent_name]["g2"].append(g2_count)
        self.eval_violations_details[agent_name]["g3"].append(g3_count)
        self.eval_violations_details[agent_name]["g4"].append(g4_count)
        self.eval_violations_details[agent_name]["g5"].append(g5_count)


class Plotter:
    """Static utility class for generating matplotlib visualizations."""

    @staticmethod
    def plot_training_results(training_log, agent_name, window=500):
        """Plots the moving average of cumulative rewards during training.

        Args:
            training_log (dict): Dictionary mapping agent names to reward lists.
            agent_name (str): The specific agent to plot.
            window (int, optional): Window size for the moving average. Defaults to 500.
        """
        rewards = training_log.get(agent_name, [])
        if len(rewards) < 50:
            return

        plt.figure(figsize=(10, 6))
        
        # Plot raw rewards with light transparency
        plt.plot(rewards, alpha=0.12, color='tab:blue')

        # Plot 50-episode MA (short-term trend) as a dotted line
        if len(rewards) >= 50:
            mv_avg_50 = pd.Series(rewards).rolling(window=50).mean()
            plt.plot(mv_avg_50, label=f"{agent_name} (MA 50)", color='tab:blue', linestyle=':', alpha=0.6)

        # Plot larger-window MA (long-term trend) as a solid line
        if len(rewards) >= window:
            mv_avg_large = pd.Series(rewards).rolling(window=window).mean()
            plt.plot(mv_avg_large, label=f"{agent_name} (MA {window})", color='tab:blue', linestyle='-', alpha=1.0)
            valid_mv_avg = mv_avg_large.dropna()
        else:
            mv_avg_large = pd.Series(rewards).rolling(window=len(rewards)).mean()
            valid_mv_avg = mv_avg_large.dropna()

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
        """Generates a bar chart comparing total training violations across agents.

        Args:
            logger (DataLogger): The logging instance containing the metrics.
        """
        agents = ["Standard RL", "safe_RL agent"]
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
    def plot_comprehensive_evaluation(logger, agent_name):
        """Generates a suite of 7 detailed plots from evaluation rollouts.

        Produces plots for nitrate trajectory, constraint violation breakdowns,
        phycocyanin production, light intensity control, nitrate feed control,
        reactor volume, and the product/biomass ratio.

        Args:
            logger (DataLogger): The logging instance containing the metrics.
            agent_name (str): The specific agent to visualize.
        """
        eval_data = logger.eval_data
        eval_violations = logger.eval_violations_details

        data = eval_data.get(agent_name)
        if data is None or "states" not in data or len(data["states"]) == 0:
            print("Evaluation data missing. Cannot plot.")
            return

        time = np.arange(len(data["states"]))
        color = 'tab:blue' if agent_name == "Standard RL" else 'tab:orange'

        # Constants
        I_MIN, I_MAX = 120.0, 400.0
        FN_MAX_GROWTH = 40.0
        FN_MAX_PROD   = 10.0
        N_LIMIT_PATH  = 800.0
        N_LIMIT_TERM  = 150.0
        RATIO_LIMIT   = 0.011
        V_MAX         = 50.0
        V_MIN         = 5.0
        
        os.makedirs("plot", exist_ok=True)

        # 1. Nitrate concentration
        plt.figure(figsize=(8, 5))
        n_best = data["states"][:, 1] * 800.0
        plt.plot(time, n_best, label="Best Run", color=color)
        agg = data.get("agg_data", {})
        if agg:
            plt.fill_between(time, agg["nitrate_min"] * 800.0, agg["nitrate_max"] * 800.0,
                             color=color, alpha=0.2, label="Min/Max")
        plt.axhline(y=N_LIMIT_PATH, color='r', linestyle='--', alpha=0.5, label="$g_1$")
        plt.axhline(y=N_LIMIT_TERM, color='darkred', linestyle='--', alpha=0.5, label="$g_3$")
        plt.title(f"Nitrate ($c_N$) — {agent_name}")
        plt.ylabel("mg/L")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.2)
        plt.legend(fontsize=9)
        plt.savefig(os.path.join("plot", "plot_nitrate.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. g1,2,3,4,5 violated during evaluation as a bar chart
        plt.figure(figsize=(8, 5))
        v_details = eval_violations.get(agent_name, {})
        g_labels = ["g1", "g2", "g3", "g4", "g5"]
        g_counts = [np.sum(v_details.get(g, [])) for g in g_labels]
        bars = plt.bar(g_labels, g_counts, color=color)
        plt.title(f"Evaluation Violations — {agent_name}")
        plt.ylabel("Count")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
        plt.savefig(os.path.join("plot", "plot_violations.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Final phycocyanin produced in concentration as well as mg
        plt.figure(figsize=(8, 5))
        if agg:
            c_q = agg["production_avg"] * 0.2
            mass_mg = c_q * (agg["volume_avg"] * V_MAX) * 1000.0
            
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(time, c_q, label="Concentration (g/L)", color=color)
            ax1.set_ylabel("Concentration (g/L)", color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xlabel("Step")
            
            ax2 = ax1.twinx()
            ax2.plot(time, mass_mg, label="Total Mass (mg)", color='tab:green', linestyle='--')
            ax2.set_ylabel("Total Mass (mg)", color='tab:green')
            ax2.tick_params(axis='y', labelcolor='tab:green')
            
            plt.title(f"Average Phycocyanin Production — {agent_name}")
            fig.tight_layout()
            plt.savefig(os.path.join("plot", "plot_phycocyanin.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.close()

        # 4. Light intensity control plot for the best episode
        plt.figure(figsize=(8, 5))
        best_I = I_MIN + ((data["actions"][:, 1] + 1.0) / 2.0) * (I_MAX - I_MIN)
        t_act = np.arange(len(best_I) + 1)
        plt.step(t_act, np.append(best_I, best_I[-1]), where='post', label="Best Run", color=color)
        plt.title(f"Light Intensity ($I$) — {agent_name}")
        plt.ylabel(r"$\mu mol/m^2/s$")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.savefig(os.path.join("plot", "plot_light.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Nitrate feed control plot for the best episode
        plt.figure(figsize=(8, 5))
        best_Fn = ((data["actions"][:, 2] + 1.0) / 2.0) * FN_MAX_GROWTH
        t_act = np.arange(len(best_Fn) + 1)
        plt.step(t_act, np.append(best_Fn, best_Fn[-1]), where='post', label="Best Run", color=color)
        plt.title(f"Nitrate Feed ($F_N$) — {agent_name}")
        plt.ylabel("mg/L/h")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.savefig(os.path.join("plot", "plot_nitrate_feed.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Reactor volume with min and max shadow area
        plt.figure(figsize=(8, 5))
        v_best = data["states"][:, 3] * V_MAX
        plt.plot(time, v_best, label="Best Run", color=color)
        if agg and "volume_min" in agg:
            plt.fill_between(time, agg["volume_min"] * V_MAX, agg["volume_max"] * V_MAX,
                             color=color, alpha=0.2, label="Min/Max")
        plt.axhline(y=V_MAX, color='r', linestyle='--', alpha=0.5, label="$g_4$")
        plt.axhline(y=V_MIN, color='darkred', linestyle='--', alpha=0.5, label="$g_5$")
        plt.title(f"Reactor Volume — {agent_name}")
        plt.ylabel("L")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.2)
        plt.legend(fontsize=9)
        plt.savefig(os.path.join("plot", "plot_volume.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 7. Cq/Cx with min and max shadow area
        plt.figure(figsize=(8, 5))
        if agg:
            plt.fill_between(time, agg["ratio_min"], agg["ratio_max"], color=color, alpha=0.2, label="Min/Max")
            plt.plot(time, agg["ratio_avg"], color=color, label="Mean")
        plt.axhline(y=RATIO_LIMIT, color='r', linestyle='--', alpha=0.5, label="$g_2$")
        plt.title(f"$c_q / c_x$ Ratio — {agent_name}")
        plt.ylabel("Ratio")
        plt.xlabel("Step")
        plt.grid(True, alpha=0.2)
        plt.legend(fontsize=9)
        plt.savefig(os.path.join("plot", "plot_ratio.png"), dpi=300, bbox_inches='tight')
        plt.close()
