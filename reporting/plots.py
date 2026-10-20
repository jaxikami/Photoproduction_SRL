"""
Visualization of training and evaluation results.

The Plotter class generates comparative visualizations of agent performance
and safety integrity from a DataLogger (reporting/logger.py) and the
evaluation trajectories saved under policy/.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reporting.paths import agent_plot_dir, compare_plot_dir

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


class Plotter:
    """Static utility class for generating matplotlib visualizations.

    Provides functions to visualize agent rewards convergence, aggregate training
    violations, and plot detailed state and control profiles across evaluation runs.
    """

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

        # Divide by max reward so the plot goes up to 1
        max_reward = max(rewards)
        if max_reward <= 0:
            max_reward = 1.0  # fallback
            
        rewards = [r / max_reward for r in rewards]

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
            y_max = 1.0
            y_min = max(-0.2, valid_mv_avg.min() - 0.1) # Clip bottom to avoid squishing the positive convergence
            plt.ylim(y_min, 1.1)

        plt.title(f"Training Convergence: {agent_name}")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward (Normalized)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(agent_plot_dir(agent_name), f"training_{agent_name}.png"), dpi=300, bbox_inches='tight')
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
        plt.savefig(os.path.join(compare_plot_dir(), "training_violations.png"), dpi=300, bbox_inches='tight')
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

        CONTROL_INTERVAL = 10.0
        suffix = "_safe" if "safe" in agent_name.lower() else "_standard"
        time_hours = np.arange(len(data["states"])) * CONTROL_INTERVAL
        color = 'tab:blue' if agent_name == "Standard RL" else 'tab:orange'

        # Constants
        I_MIN, I_MAX = 120.0, 400.0
        FN_MAX_GROWTH = 40.0
        FN_MAX_PROD   = 10.0
        N_LIMIT_PATH  = 800.0
        N_LIMIT_TERM  = 150.0
        RATIO_LIMIT   = 0.011
        V_MAX         = 20.0
        V_MIN         = 2.0
        
        out_dir = agent_plot_dir(agent_name)

        # 1. Nitrate concentration
        plt.figure(figsize=(8, 5))
        n_best = data["states"][:, 1] * 800.0
        plt.plot(time_hours, n_best, label="Best Run", color=color)
        agg = data.get("agg_data", {})
        if agg:
            plt.fill_between(time_hours, agg["nitrate_min"] * 800.0, agg["nitrate_max"] * 800.0,
                             color=color, alpha=0.2, label="Min/Max")
        plt.axhline(y=N_LIMIT_PATH, color='r', linestyle='--', alpha=0.5, label="$g_1$")
        plt.axhline(y=N_LIMIT_TERM, color='darkred', linestyle='--', alpha=0.5, label="$g_3$")
        plt.title(f"Nitrate ($c_N$) — {agent_name}")
        plt.ylabel("mg/L")
        plt.xlabel("Time (hours)")
        plt.grid(True, alpha=0.2)
        plt.legend(fontsize=9)
        plt.savefig(os.path.join(out_dir, f"plot_nitrate{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. g1,2,3,4,5 violated during evaluation as a bar chart
        plt.figure(figsize=(8, 5))
        v_details = eval_violations.get(agent_name, {})
        g_labels = ["g1", "g2", "g3", "g4", "g5"]
        g_counts = [np.sum(np.array(v_details.get(g, [])) > 0) for g in g_labels]
        bars = plt.bar(g_labels, g_counts, color=color)
        plt.title(f"Evaluation Violations — {agent_name}")
        plt.ylabel("Count")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
        plt.savefig(os.path.join(out_dir, f"plot_violations{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Final phycocyanin produced in concentration as well as g
        plt.figure(figsize=(8, 5))
        if agg:
            c_q = agg["production_avg"] * 0.2
            # Use cumulative harvested mass from aggregation metrics
            if "harvested_avg" in agg:
                mass_g = agg["harvested_avg"]
            else:
                mass_g = c_q * (agg["volume_avg"] * V_MAX)
            
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(time_hours, c_q, label="Concentration (g/L)", color=color)
            ax1.set_ylabel("Concentration (g/L)", color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xlabel("Time (hours)")
            
            ax2 = ax1.twinx()
            ax2.plot(time_hours, mass_g, label="Total Mass (g)", color='tab:green', linestyle='--')
            ax2.set_ylabel("Total Mass (g)", color='tab:green')
            ax2.tick_params(axis='y', labelcolor='tab:green')
            
            plt.title(f"Average Phycocyanin Production — {agent_name}")
            fig.tight_layout()
            plt.savefig(os.path.join(out_dir, f"plot_phycocyanin{suffix}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.close()

        # 4. Light intensity control plot for the best episode
        plt.figure(figsize=(8, 5))
        
        stages = data["metrics"]["current_stage"].values
        nitrate_supplies = data["metrics"]["nitrate_supply"].values
        
        best_I = []
        best_Fn = []
        for idx in range(len(data["actions"])):
            action = data["actions"][idx]
            stage = stages[idx]
            n_supply = nitrate_supplies[idx]
            
            a_scaled = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
            
            if stage in (0, 1):
                I_val = I_MIN + a_scaled[1] * (I_MAX - I_MIN)
                fn_max = FN_MAX_GROWTH if stage == 0 else FN_MAX_PROD
                Fn_val = a_scaled[2] * fn_max
            elif stage == 2:
                I_val = I_MIN
                Fn_val = 0.0
            else:
                I_val = I_MIN
                Fn_val = 0.0
                
            if n_supply <= 0:
                Fn_val = 0.0
                
            best_I.append(I_val)
            best_Fn.append(Fn_val)
            
        best_I = np.array(best_I)
        best_Fn = np.array(best_Fn)

        t_act = np.arange(len(best_I) + 1) * CONTROL_INTERVAL
        plt.step(t_act, np.append(best_I, best_I[-1]), where='post', label="Best Run", color=color)
        plt.title(f"Light Intensity ($I$) — {agent_name}")
        plt.ylabel(r"$\mu mol/m^2/s$")
        plt.xlabel("Time (hours)")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"plot_light{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Nitrate feed control plot for the best episode
        plt.figure(figsize=(8, 5))
        t_act = np.arange(len(best_Fn) + 1) * CONTROL_INTERVAL
        plt.step(t_act, np.append(best_Fn, best_Fn[-1]), where='post', label="Best Run", color=color)
        plt.title(f"Nitrate Feed ($F_N$) — {agent_name}")
        plt.ylabel("mg/L/h")
        plt.xlabel("Time (hours)")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"plot_nitrate_feed{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Reactor volume with min and max shadow area
        plt.figure(figsize=(8, 5))
        v_best = data["states"][:, 3] * V_MAX
        plt.plot(time_hours, v_best, label="Best Run", color=color)
        if agg and "volume_min" in agg:
            plt.fill_between(time_hours, agg["volume_min"] * V_MAX, agg["volume_max"] * V_MAX,
                             color=color, alpha=0.2, label="Min/Max")
        plt.axhline(y=V_MAX, color='r', linestyle='--', alpha=0.5, label="$g_4$")
        plt.axhline(y=V_MIN, color='darkred', linestyle='--', alpha=0.5, label="$g_5$")
        plt.title(f"Reactor Volume — {agent_name}")
        plt.ylabel("L")
        plt.xlabel("Time (hours)")
        plt.grid(True, alpha=0.2)
        plt.legend(fontsize=9)
        plt.savefig(os.path.join(out_dir, f"plot_volume{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 7. Cq/Cx with min and max shadow area
        plt.figure(figsize=(8, 5))
        if agg:
            plt.fill_between(time_hours, agg["ratio_min"], agg["ratio_max"], color=color, alpha=0.2, label="Min/Max")
            plt.plot(time_hours, agg["ratio_avg"], color=color, label="Mean")
        plt.axhline(y=RATIO_LIMIT, color='r', linestyle='--', alpha=0.5, label="$g_2$")
        plt.title(f"$c_q / c_x$ Ratio — {agent_name}")
        plt.ylabel("Ratio")
        plt.xlabel("Time (hours)")
        plt.grid(True, alpha=0.2)
        plt.legend(fontsize=9)
        plt.savefig(os.path.join(out_dir, f"plot_ratio{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_comparison_table():
        """Generates a table plot comparing final production and concentration for both RL agents."""
        safe_path = os.path.join("policy", "eval_data_safe.npz")
        std_path = os.path.join("policy", "eval_data_standard.npz")
        
        data_safe = None
        data_std = None
        
        if os.path.exists(safe_path):
            data_safe = np.load(safe_path)
        if os.path.exists(std_path):
            data_std = np.load(std_path)
            
        if data_safe is None and data_std is None:
            print("No evaluation data found for comparison table.")
            return
            
        rows = []
        cell_text = []
        
        if data_std is not None:
            mass = data_std["final_mass"] if "final_mass" in data_std else data_std['states'][-1, 2] * 0.2 * data_std['states'][-1, 3] * 20.0
            conc = data_std["final_conc"] if "final_conc" in data_std else data_std['states'][-1, 2] * 0.2
            rows.append("Standard RL")
            cell_text.append([f"{float(mass):.2f}", f"{float(conc):.4f}"])
            
        if data_safe is not None:
            mass = data_safe["final_mass"] if "final_mass" in data_safe else data_safe['states'][-1, 2] * 0.2 * data_safe['states'][-1, 3] * 20.0
            conc = data_safe["final_conc"] if "final_conc" in data_safe else data_safe['states'][-1, 2] * 0.2
            rows.append("Safe RL")
            cell_text.append([f"{float(mass):.2f}", f"{float(conc):.4f}"])
            
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=cell_text,
                         rowLabels=rows,
                         colLabels=["Final Production (g)", "Final Concentration (g/L)"],
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title("Comparison of Final Production and Concentration")
        plt.savefig(os.path.join(compare_plot_dir(), "comparison_table.png"), dpi=300, bbox_inches='tight')
        plt.close()
