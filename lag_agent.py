
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# Hardware setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCriticStandardRL(nn.Module):
    """
    Standard PPO Actor-Critic for the multi-stage photobioreactor.

    Generates raw action intents in a 4D action space:
      [time_multiplier, light_intensity, nitrate_feed, outstream_flow]

    Observation is 11D (see env.py get_state_norm for layout).
    """
    def __init__(self, state_dim=11, action_dim=4):
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
        """
        Generates an action during rollout.

        Returns:
            z:        squashed action in [-1, 1]
            log_prob: log probability of the action
            z_raw:    unbounded Gaussian sample (for PPO updates)
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
        """
        Re-evaluates stored actions under the current policy for the PPO update.
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
    """
    PPO wrapper for the standard (unconstrained) RL agent.
    """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coeff):
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
        """
        Selects an action using the old policy for trajectory collection.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm).to(device).unsqueeze(0)
            z, log_prob, z_raw = self.policy_old.act(state_t)

        return z.cpu().numpy().flatten(), log_prob.cpu().numpy(), z_raw.cpu().numpy().flatten()

    def learn(self, memory):
        """
        PPO update with clipped surrogate objective.
        """
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_z_raw = torch.squeeze(torch.stack(memory.raw_actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_z_raw)

            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach().squeeze()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            ppo_loss = -torch.min(surr1, surr2).mean()

            loss = ppo_loss + \
                   0.5 * self.MseLoss(state_values.squeeze(), rewards) - \
                   self.entropy_coeff * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())