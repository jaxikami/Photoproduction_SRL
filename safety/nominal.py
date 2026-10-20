"""The nominal plant: envs/plant.py's own step, run from an observation.

The state estimator's model of the photobioreactor (safety/estimator.py), as
Fermentation-PPO's estimator uses safety/plant_sim.py.  The kinetics, credits,
stage logic and nitrate bookkeeping are PhycocyaninEnvCore._physics_step
itself, so on the undisturbed plant a prediction from the true state is exact.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from envs.plant import PhycocyaninEnvCore


class NominalPlant(PhycocyaninEnvCore):
    """PhycocyaninEnvCore whose physical state is set from a normalised observation."""

    def load(self, obs):
        """Set the plant to the state an observation describes.

        Args:
            obs: Normalised observation [12] (get_state_norm()'s layout).

        Returns:
            self.
        """
        obs = np.asarray(obs, dtype=np.float64)
        self.state = np.array([obs[0] * 6.0, obs[1] * 800.0, obs[2] * 0.2, obs[3] * self.V_MAX])
        self.current_stage = int(np.argmax(obs[4:8]))
        self.stage_credits = float(obs[8] * self.BASE_CREDITS[self.current_stage])
        self.time = float(obs[9] * self.total_time)
        self.time_step_count = int(round(self.time / self.control_freq))
        self.nitrate_supply = float(obs[10] * self.INITIAL_NITRATE_SUPPLY)
        # The Harvesting latch is set at the start of a step below V_DRAIN, and
        # volume only falls in Harvesting: the step re-derives it from V.
        self._cleanup_latch = False
        return self


_PLANT = NominalPlant()


def predict(obs, action):
    """The nominal plant's next observation [12] after holding ``action`` from ``obs``."""
    p = _PLANT.load(obs)
    p._physics_step(np.asarray(action, dtype=np.float64))
    return p.get_state_norm()
