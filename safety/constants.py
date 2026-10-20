"""Constants shared by the APN's dataset, training, projection and validation.

The physical values mirror envs/plant.py; the APN is trained on its own copy
so the safety stack can be checked against the plant it guards.
"""

import math
import os

# ── Physical parameters (mirror envs/plant.py) ───────────────────────────────
I_MIN, I_MAX     = 120.0, 400.0
FN_MAX_GROWTH    = 40.0
FN_MAX_PROD      = 10.0
FOUT_MAX         = 2.0
N_LIMIT_PATH     = 800.0
RATIO_LIMIT      = 0.011

V_MAX            = 20.0
V_MIN            = 2.0
V_RESET          = 0.10 * V_MAX   # 2.0 L — post-harvest partial reset (10% V_MAX)
C_N_STOCK        = 3000.0
CONTROL_INTERVAL = 10.0
TOTAL_TIME       = 1000.0
SAFE_BUFFER      = 0.98

# ── G3 (nitrate at the Growth -> Harvesting transition) ──────────────────────
N_LIMIT_TERM     = 150.0
# G3's own safety buffer: the label's target is N_LIMIT_TERM * G3_BUFFER.  Lower
# it if the G3 validation test overshoots (the head is learned, so it errs).
G3_BUFFER        = SAFE_BUFFER
BASE_CREDITS     = (163.0, 81.0)             # Inoculation, Growth credit-hours
INITIAL_NITRATE_SUPPLY = 100000.0
MULT_MIN, MULT_RANGE   = 0.592, 1.376        # time multiplier = MULT_MIN + a_scaled * MULT_RANGE
# The G3 fallback: the fastest possible nitrate draw-down.  Nitrate only falls
# through uptake, so no feed, the slowest clock (most time before Growth ends)
# and the light that maximises growth (Aiba optimum sqrt(ks * ki)).
I_OPT            = math.sqrt(178.9 * 447.1)  # ~282.8 umol/m^2/s
# The fallback as a squashed action (time multiplier, light, feed, outflow).
G3_FALLBACK      = (-1.0, 2.0 * (I_OPT - I_MIN) / (I_MAX - I_MIN) - 1.0, -1.0, -1.0)
G3_DT            = 1.0                       # RK4 step (h) of the label simulation

# ── Projection (L2) ───────────────────────────────────────────────────────────
THRESHOLD        = 0.711   # safety probability an action is projected up to
MAX_PROJ_STEPS   = 5
LR_PROJ          = 0.5

# The margin logit at THRESHOLD: the safety gradient's hinge target, so the
# actor is pushed to exactly where the projection would stop correcting.
SAFE_LOGIT       = math.log(THRESHOLD / (1.0 - THRESHOLD))

# ── Checkpoint ────────────────────────────────────────────────────────────────
APN_PATH         = os.path.join("policy", "action_projection_network.pth")
