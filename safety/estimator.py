"""The state estimate the APN acts on (as Fermentation-PPO's safety/estimator.py).

The APN's margins are only as good as the state they are computed from.  Under
sensor noise the raw reading of nitrate or volume can sit several percent away
from the truth, which is wider than the heads' static buffers (2%): the
projection then trusts a reading that is too low and lets the plant overshoot.
StateEstimator filters the measured features before the APN sees them:

    predict    the nominal plant (safety/nominal.py, the environment's own
               step) holds the action actually sent for one decision, from
               the last estimate and from the last raw reading.
    correct    a scalar Kalman update per feature, with the measurement noise
               R and the model mismatch Q identified online from the plant's
               own readings.

Sensor noise is relative (a reading is the value times 1 + noise), so R and Q
are identified as fractions of the feature and scaled by its current value.
R comes from the one-step prediction error of the raw readings,
e_t = z_t - f(z_{t-1}): sensor noise makes consecutive errors anti-correlated,
E[e_t e_{t-1}] = -R (the same reading enters both), while an actuator error or
any other process noise is white and leaves that product at 0, so it is not
mistaken for sensor noise.  Until enough of those products are in, R is the
upper bound E[e_t^2] / 2 instead (e_t holds the sensor noise of two readings),
so a sensor that turns out noisy is filtered from its second reading on.  Q is
then set so the filter's own innovations nu = z - prior stay consistent,
E[nu^2] = P + Q + R (the adaptive-Kalman rule): on the nominal plant the model
is exact and Q stays at its floor, an actuator error inflates the innovations
and with them Q and the gain.  Without sensor noise R is 0 and the gain is 1:
the estimate is the reading, so the APN behaves exactly as without the filter.
A stage the model did not predict restarts the filter from the reading.

The posterior standard deviation of every feature is returned too, so the APN
can be shown a state moved towards every constraint by the estimate's own
uncertainty (pessimistic()), as Fermentation widens its buffers by it.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
from collections import deque

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from safety.nominal import predict

# Filtered features: Cx, CN, Cq, V, stage credit, time, nitrate supply and
# time left.  The stage one-hot is snapped to its largest entry.
DIMS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
# Floor of the scale R and Q are relative to (normalised units).
_SCALE_FLOOR = 1e-3
# Relative noise variance below which a feature counts as noise-free.
_R_ZERO = 1e-10
# Lag-1 products needed before R is read from them (before: the E[e^2] bound).
_MIN_PRODUCTS = 20
# Median of a chi-square(1) variable: median(x^2) / 0.455 estimates E[x^2].
_CHI2_MEDIAN = 0.455
# Pessimistic direction of each filtered feature for the APN's heads: less
# biomass (less nitrate uptake, higher Cq/Cx), more nitrate, product and volume.
_PESSIMISM = {0: -1.0, 1: 1.0, 2: 1.0, 3: 1.0}


def _stage(obs):
    """Stage index of an observation (largest one-hot entry)."""
    return int(np.argmax(obs[4:8]))


def _scale2(x):
    """Squared scale of each feature that relative variances multiply."""
    return np.maximum(np.abs(x), _SCALE_FLOOR) ** 2


class StateEstimator:
    """Kalman filter of the measured features around the nominal plant.

    Attributes:
        window: Decisions of prediction errors kept for the noise estimates.
        q_floor: Minimum relative model-mismatch std per decision.
        min_q_steps: Decisions of innovations needed before Q leaves its floor.
        x: Current estimate [12] (None before the first reading).
        std: Posterior std of the estimate [12] (0 on unfiltered dims).
    """

    def __init__(self, window=300, q_floor=2e-3, min_q_steps=8):
        self.window = window
        self.q_floor = q_floor
        self.min_q_steps = min_q_steps
        self.reset(sensors=True)

    def reset(self, sensors=False):
        """Forget the plant's state (a new episode).

        Args:
            sensors: Also forget the noise estimates (sensor noise and model
                mismatch).  They belong to the sensors and actuators, not to
                an episode, so by default they carry over from one episode to
                the next.
        """
        if sensors:
            self._hist_ee = deque(maxlen=self.window)
            self._hist_e2 = deque(maxlen=self.window)
            self._hist_nu = deque(maxlen=self.window)
        self.x = None
        self.std = None
        self._P = None
        self._prior = None
        self._raw_prior = None
        self._z_prev = None
        self._e_prev = None

    # ── Noise identification ─────────────────────────────────────────────────

    def _noise(self):
        """Relative measurement noise R and model mismatch Q per feature, [F] each."""
        if len(self._hist_ee) >= _MIN_PRODUCTS:
            r = np.maximum(-np.mean(np.array(self._hist_ee), axis=0), 0.0)
        elif self._hist_e2:
            r = np.mean(np.array(self._hist_e2), axis=0) / 2.0
        else:
            r = np.zeros(len(DIMS))
        r = np.where(r < _R_ZERO, 0.0, r)
        q = np.full_like(r, self.q_floor ** 2)
        if len(self._hist_nu) >= self.min_q_steps:
            # Innovation excess over what the filter expects without Q.
            excess = np.median(np.array(self._hist_nu), axis=0) / _CHI2_MEDIAN - r
            q = np.maximum(excess, q)
        return r, q

    # ── Filter ───────────────────────────────────────────────────────────────

    def update(self, z):
        """Correct the estimate with a new reading.

        Args:
            z: Observation [12] (possibly noisy).

        Returns:
            A tuple of the estimate [12] (float32) and its posterior std [12]
            (0 on the dims that are not filtered).
        """
        z = np.asarray(z, dtype=np.float64)
        zf = z[DIMS]
        same = self._prior is not None and _stage(self._prior) == _stage(z)
        if same:
            prior = self._prior[DIMS]
            # Relative prediction errors of the raw readings; their lag-1
            # product is -R under sensor noise and 0 under process noise.
            e = (zf - self._raw_prior[DIMS]) / np.sqrt(_scale2(self._raw_prior[DIMS]))
            self._hist_e2.append(e ** 2)
            if self._e_prev is not None:
                self._hist_ee.append(e * self._e_prev)
            self._e_prev = e
            s2 = _scale2(prior)
            self._hist_nu.append(((zf - prior) ** 2 - self._P) / s2)
            r, q = self._noise()
            R, P_pri = r * s2, self._P + q * s2
            K = np.where(R > 0.0, P_pri / np.maximum(P_pri + R, 1e-300), 1.0)
            xf = zf - (1.0 - K) * (zf - prior)
            P = (1.0 - K) * P_pri
        else:
            # First reading, or a stage the model did not predict: the
            # reading itself, as uncertain as the sensors are.
            self._e_prev = None
            r, _ = self._noise()
            R = r * _scale2(zf)
            xf, P = zf.copy(), R.copy()
        self._P = P
        self._z_prev = z

        x = z.copy()
        x[DIMS] = np.maximum(xf, 0.0)
        x[4:8] = 0.0
        x[4 + _stage(z)] = 1.0
        std = np.zeros_like(z)
        # Never less certain than the reading itself.
        std[DIMS] = np.sqrt(np.minimum(P, R))
        self.x, self.std = x, std
        return x.astype(np.float32), std.astype(np.float32)

    def process_std(self):
        """How far one decision's outcome strays from the nominal plant, [12].

        The model-mismatch std Q above its floor (an actuator error, a
        disturbance), in normalised units at the current estimate; 0 on the
        nominal plant and before any reading.
        """
        std = np.zeros(12)
        if self.x is None:
            return std
        _, q = self._noise()
        excess = np.maximum(q - self.q_floor ** 2, 0.0)
        std[DIMS] = np.sqrt(excess * _scale2(self.x[DIMS]))
        return std

    def predict(self, action):
        """Predict the next reading from the estimate and from the last raw reading.

        Call after the action is chosen (the action actually sent).

        Args:
            action: Action [4] in [-1, 1] (stage-masked, projected).
        """
        if self.x is None:
            return
        self._prior = predict(self.x, action)
        self._raw_prior = predict(self._z_prev, action)


def pessimistic(x, std, kappa):
    """The estimate moved ``kappa`` standard deviations towards every constraint.

    Args:
        x: State estimate [12].
        std: Its posterior std [12].
        kappa: Standard deviations of the move.

    Returns:
        Observation [12] (float32): less biomass, more nitrate, product and
        volume; x itself where std is 0.
    """
    out = np.array(x, dtype=np.float32)
    for d, sign in _PESSIMISM.items():
        out[d] = max(out[d] + sign * kappa * std[d], 0.0)
    return out
