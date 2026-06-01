"""
Standard Proximal Policy Optimization (PPO) Agent.

This module implements the benchmark PPO agent used for training the
unconstrained or penalty-based formulation of the photobioreactor environment.
It includes the Actor-Critic network definition and the PPO update loop.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from numba import njit

# Hardware setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@njit(cache=True)
def _discount_rewards(raw_rewards, terminals, gamma):
    """Numba-accelerated reward discounting."""
    n = len(raw_rewards)
    discounted = np.zeros(n, dtype=np.float32)
    running = 0.0
    for i in range(n - 1, -1, -1):
        if terminals[i]:
            running = 0.0
        running = raw_rewards[i] + gamma * running
        discounted[i] = running
    return discounted


class ActorCriticStandardRL(nn.Module):
    """Standard PPO Actor-Critic for the multi-stage photobioreactor.

    This neural network architecture maps the 12D continuous state space
    to a 4D continuous action space. It uses a Gaussian policy where the
    actor outputs the mean and a trainable parameter determines the log standard
    deviation. The critic outputs the state-value estimate.

    Observation space (12D):
        Defined by `env.get_state_norm()`. Includes physical states, one-hot
        encoded operational stages, and remaining time/nutrient credits.

    Action space (4D):
        [time_multiplier, light_intensity, nitrate_feed, outstream_flow].
        The outputs are squashed to [-1, 1] using tanh.
    """
    def __init__(self, state_dim=12, action_dim=4):
        """Initializes the Actor and Critic neural networks.

        Args:
            state_dim (int, optional): Dimension of the observation space. Defaults to 12.
            action_dim (int, optional): Dimension of the action space. Defaults to 4.
        """
        super(ActorCriticStandardRL, self).__init__()
        self.LOG_STD_MIN = -1.0
        self.LOG_STD_MAX = 0.5

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state):
        """Generates an action during environment rollout.

        Samples an unbounded action from the Gaussian policy, computes its log 
        probability, and squashes the action to [-1, 1] using tanh. The log 
        probability is corrected for the tanh squashing function.

        Args:
            state (torch.Tensor): The current state observation tensor.

        Returns:
            tuple:
                - z (torch.Tensor): Squashed action in [-1, 1] applied to the env.
                - log_prob (torch.Tensor): Log probability of the chosen action.
                - z_raw (torch.Tensor): Unbounded Gaussian sample stored for PPO updates.
        """
        mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)

        z_raw = dist.sample()
        z = torch.tanh(z_raw)

        log_prob = dist.log_prob(z_raw).sum(dim=-1)
        log_prob -= torch.log(1 - z.pow(2) + 1e-6).sum(dim=-1)

        return z.detach(), log_prob.detach(), z_raw.detach()

    def evaluate(self, state, z_raw):
        """Re-evaluates stored actions under the current policy for the PPO update.

        Args:
            state (torch.Tensor): Batch of state observations.
            z_raw (torch.Tensor): Batch of unbounded actions previously taken.

        Returns:
            tuple:
                - log_probs (torch.Tensor): New log probabilities of the actions.
                - state_values (torch.Tensor): Critic's state-value estimates.
                - dist_entropy (torch.Tensor): Entropy of the current action distribution.
        """
        mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mean, std)

        log_probs = dist.log_prob(z_raw).sum(dim=-1)
        z_tanhed = torch.tanh(z_raw)
        log_probs -= torch.log(1 - z_tanhed.pow(2) + 1e-6).sum(dim=-1)

        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        return log_probs, state_values, dist_entropy


class StandardRL_Agent:
    """PPO algorithm wrapper for the standard (penalty-based) RL agent.
    
    This class handles trajectory collection, reward discounting, advantage
    normalization, and the core PPO surrogate loss optimization loop.
    """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coeff):
        """Initializes the PPO agent, networks, and optimizers.

        Args:
            state_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            lr_actor (float): Learning rate for the actor network.
            lr_critic (float): Learning rate for the critic network.
            gamma (float): Discount factor for future rewards.
            K_epochs (int): Number of optimization epochs per PPO update.
            eps_clip (float): PPO clipping parameter for the surrogate objective.
            entropy_coeff (float): Coefficient for the entropy bonus to encourage exploration.
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coeff = entropy_coeff

        self.policy = ActorCriticStandardRL(state_dim=state_dim, action_dim=action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': [self.policy.log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCriticStandardRL(state_dim=state_dim, action_dim=action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state_norm):
        """Selects an action using the old (frozen) policy during trajectory collection.

        Args:
            state_norm (np.ndarray): The normalized state observation.

        Returns:
            tuple:
                - z (np.ndarray): The squashed action array.
                - log_prob (np.ndarray): Log probability of the action.
                - z_raw (np.ndarray): Unbounded raw action array.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm).to(device).unsqueeze(0)
            z, log_prob, z_raw = self.policy_old.act(state_t)

        return z.cpu().numpy().flatten(), log_prob.cpu().numpy(), z_raw.cpu().numpy().flatten()

    def learn(self, memory):
        """Executes the PPO optimization update.

        Calculates discounted rewards, normalizes advantages, and performs
        `K_epochs` of mini-batch gradient descent to update the actor and critic
        networks using the PPO clipped surrogate loss.

        Args:
            memory (Memory): Buffer containing states, actions, rewards, and logprobs
                collected during the recent trajectory rollout.
        """
        # Fast vectorized reward discounting using numba
        if hasattr(memory, '_rewards'):
            raw_rewards = memory._rewards[:memory._ptr].copy()
            terminals = memory._is_terminals[:memory._ptr].astype(np.float32)
        else:
            raw_rewards = np.array(memory.rewards, dtype=np.float32)
            terminals = np.array(memory.is_terminals, dtype=np.float32)
        discounted = _discount_rewards(raw_rewards, terminals, self.gamma)

        rewards = torch.from_numpy(discounted).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Use fast pre-stacked tensor path if available
        if hasattr(memory, 'get_tensors'):
            old_states, _, old_z_raw, old_logprobs, _ = memory.get_tensors(device)
        else:
            old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
            old_z_raw = torch.squeeze(torch.stack(memory.raw_actions, dim=0)).detach().to(device)
            old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        # Precalculate and normalize advantages over the entire batch
        with torch.no_grad():
            _, old_state_values, _ = self.policy.evaluate(old_states, old_z_raw)
            old_state_values = old_state_values.squeeze(-1)
            advantages = rewards - old_state_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = old_states.size(0)
        mini_batch_size = 256

        for _ in range(self.K_epochs):
            permutation = torch.randperm(batch_size)
            for start_idx in range(0, batch_size, mini_batch_size):
                batch_indices = permutation[start_idx : start_idx + mini_batch_size]

                b_states = old_states[batch_indices]
                b_z_raw = old_z_raw[batch_indices]
                b_logprobs = old_logprobs[batch_indices]
                b_rewards = rewards[batch_indices]
                b_advantages = advantages[batch_indices]

                logprobs, state_values, dist_entropy = self.policy.evaluate(b_states, b_z_raw)

                ratios = torch.exp(logprobs - b_logprobs)

                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                ppo_loss = -torch.min(surr1, surr2).mean()

                loss = ppo_loss + \
                       0.5 * self.MseLoss(state_values.squeeze(-1), b_rewards) - \
                       self.entropy_coeff * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())