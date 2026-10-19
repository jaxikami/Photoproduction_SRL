"""Metrics tracking for training and evaluation.

DataLogger records episode rewards, constraint violations and evaluation
trajectories; reporting/plots.py turns them into figures.
"""
import numpy as np
import pandas as pd


class DataLogger:
    """Centralized logging architecture for recording agent metrics.
    
    Tracks rewards, trajectories, g1-g4 constraint violations, and idle-stage
    compliance (g5) during both training and evaluation.

    Attributes:
        training_log (dict): Map from agent names to lists of total rewards.
        training_violations (dict): Map from agent names to lists of total violations.
        eval_data (dict): Map from agent names to nested dictionaries containing detailed
            evaluation rollout results (states, actions, rewards, metrics, etc.).
        eval_violations (dict): Map from agent names to lists of total evaluation violations.
        eval_violations_details (dict): Map from agent names to dictionaries containing lists of
            specific constraint violation counts (g1, g2, g3, g4, g5).
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
