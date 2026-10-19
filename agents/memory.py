"""Rollout storage and reward normalisation for the PPO updates."""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np


# ─── Utility Classes ──────────────────────────────────────────────────────────


class RunningMeanStd:
    """Online reward-variance tracker using Welford's algorithm.

    Keeps a running mean and variance over all reward samples seen so far.
    Call ``update(x)`` each rollout, then ``normalize(x)`` to whiten rewards
    before GAE computation.  Whitened rewards reduce penalty-spike variance
    without destroying the reward signal direction.

    Attributes:
        mean: Running mean of observed samples.
        var: Running variance of observed samples.
        count: Total effective sample count (includes epsilon offset).
    """

    def __init__(self, epsilon=1e-4):
        """Initialise running statistics.

        Args:
            epsilon: Small constant added to initial count for numerical stability.
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        """Incorporate a new batch of reward samples into the running statistics.

        Args:
            x: Reward array of any shape. Flattened internally before update.
        """
        x_flat = np.asarray(x).flatten().astype(np.float64)
        n = len(x_flat)
        batch_mean = x_flat.mean()
        batch_var = x_flat.var()
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        m2 = self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x, clip=10.0):
        """Return a whitened copy of x using the running mean and variance.

        Output is clipped to [-clip, clip] to guard against outlier rewards
        from Lagrangian penalty spikes (G1, G3, G5).

        Args:
            x: Array to normalise.
            clip: Symmetric clipping bound applied after whitening.

        Returns:
            Normalised array of the same shape as x.
        """
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip)


class PPOMemory:
    """Rollout buffer that accumulates one trajectory segment for the PPO update.

    Backed by contiguous pre-allocated NumPy arrays filled by index, so a
    rollout costs no per-step allocation.  Call clear() after learn()
    consumes the buffer.

    Attributes:
        capacity: Steps the buffer holds (one update window).
        states: [capacity, state_dim] observations fed to the actor.
        actions: [capacity, action_dim] raw intents z from the actor (pre-tanh).
        safe_actions: [capacity, action_dim] actions sent to the environment.
        logprobs: [capacity] log pi_old(z | s, h).
        rewards: [capacity] step rewards.
        is_terminals: [capacity] episode-end flags.
    """

    def __init__(self, capacity, state_dim, action_dim):
        """Allocate an empty rollout buffer.

        Args:
            capacity: Maximum number of steps before learn() is called.
            state_dim: Observation width.
            action_dim: Action width.
        """
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.safe_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.logprobs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.is_terminals = np.zeros(capacity, dtype=np.bool_)
        self.size = 0

    def append(self, state, action, logprob, safe_action, reward, is_terminal):
        """Store one transition.

        Args:
            state: Observation the actor saw.
            action: Raw intent from the actor.
            logprob: Log-probability of the intent under the old policy.
            safe_action: Action sent to the environment.
            reward: Step reward.
            is_terminal: Whether this step ends the episode.
        """
        i = self.size
        self.states[i] = state
        self.actions[i] = action
        self.logprobs[i] = np.asarray(logprob).item()
        self.safe_actions[i] = safe_action
        self.rewards[i] = reward
        self.is_terminals[i] = is_terminal
        self.size += 1

    def clear(self):
        """Reset the write pointer; the arrays are reused."""
        self.size = 0
