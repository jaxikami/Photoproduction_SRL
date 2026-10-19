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

# ── Projection (L2) ───────────────────────────────────────────────────────────
THRESHOLD        = 0.711   # safety probability an action is projected up to
MAX_PROJ_STEPS   = 5
LR_PROJ          = 0.5

# The margin logit at THRESHOLD: the safety gradient's hinge target, so the
# actor is pushed to exactly where the projection would stop correcting.
SAFE_LOGIT       = math.log(THRESHOLD / (1.0 - THRESHOLD))

# ── Checkpoint ────────────────────────────────────────────────────────────────
APN_PATH         = os.path.join("policy", "action_projection_network.pth")
