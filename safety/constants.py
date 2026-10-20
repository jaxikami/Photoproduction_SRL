"""Constants shared by the APN's dataset, training and projection.

The physical values mirror envs/plant.py; the APN keeps its own copy.
"""

import math
import os

# ── Physical parameters (mirror envs/plant.py) ───────────────────────────────
I_MIN, I_MAX     = 120.0, 400.0
FN_MAX_GROWTH    = 40.0
FN_MAX_PROD      = 10.0
N_LIMIT_PATH     = 800.0
RATIO_LIMIT      = 0.011

V_MAX            = 20.0
C_N_STOCK        = 3000.0
CONTROL_INTERVAL = 10.0
TOTAL_TIME       = 1000.0
SAFE_BUFFER      = 0.98

# ── G3 (nitrate at the Growth -> Harvesting transition) ──────────────────────
N_LIMIT_TERM     = 150.0
# G3's own safety buffer: the head's target is N_LIMIT_TERM * G3_BUFFER.
G3_BUFFER        = SAFE_BUFFER
BASE_CREDITS     = (163.0, 81.0)             # Inoculation, Growth credit-hours
MULT_MIN, MULT_RANGE   = 0.592, 1.376        # time multiplier = MULT_MIN + a_scaled * MULT_RANGE
# The G3 fallback: the fastest possible nitrate draw-down.  Nitrate only falls
# through uptake, so no feed, the slowest clock (most time before Growth ends)
# and the light that maximises growth (Aiba optimum sqrt(ks * ki)).
I_OPT            = math.sqrt(178.9 * 447.1)  # ~282.8 umol/m^2/s
# The fallback as a squashed action (time multiplier, light, feed, outflow).
G3_FALLBACK      = (-1.0, 2.0 * (I_OPT - I_MIN) / (I_MAX - I_MIN) - 1.0, -1.0, -1.0)
# Credit-hours one fallback control step uses (the slowest clock).
FALLBACK_CREDITS = MULT_MIN * CONTROL_INTERVAL
# Fallback control steps after an action until Growth ends, at most: the rest
# of a fresh Inoculation, then a whole Growth (28 + 14).
DRAWDOWN_STEPS   = (math.ceil(BASE_CREDITS[0] / FALLBACK_CREDITS)
                    + math.ceil(BASE_CREDITS[1] / FALLBACK_CREDITS))

# ── Projection (L2) ───────────────────────────────────────────────────────────
PROJ_EPS         = 1e-3    # margin every corrected exact head is brought up to
# ... and the learned G3 head (1.5 mg/L): KineticNet's nitrate error near the
# G3 target is ~0.5 mg/L on average, about 1 mg/L at worst.
G3_EPS           = 1e-2
PROJ_DAMPING     = 1e-4    # base damping of the minimum-norm correction
PROJ_ITERS       = 5       # joint correction steps (re-linearised each time)
CONTAINMENT_ITERS = 2      # final steps on the containment heads alone

# ── Checkpoint ────────────────────────────────────────────────────────────────
KINETIC_PATH     = os.path.join("policy", "apn_kinetic.pth")
