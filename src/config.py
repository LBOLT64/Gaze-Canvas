"""
Gaze Canvas — module-level configuration constants.

All tuneable values live here so every other module can
``from src.config import …`` without circular dependencies.
"""

import logging

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 1280, 800

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT = 1280, 720

# ---------------------------------------------------------------------------
# Gaze smoothing  (One Euro Filter parameters)
# ---------------------------------------------------------------------------
OEF_MIN_CUTOFF = 1.7   # lower = smoother, higher = more responsive
OEF_BETA = 0.8         # higher = faster reaction to speed changes
OEF_D_CUTOFF = 1.0     # derivative low-pass cutoff

# ---------------------------------------------------------------------------
# Brush
# ---------------------------------------------------------------------------
BRUSH_MIN, BRUSH_MAX = 4, 36
BRUSH_DEFAULT = 14

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
CAL_POINTS = 9
CAL_SAMPLES_PER_POINT = 25       # frames to collect per target
CAL_OUTLIER_SIGMA = 2.0          # reject samples beyond this many σ

# ---------------------------------------------------------------------------
# Blink detection  (EAR-based)
# ---------------------------------------------------------------------------
BLINK_EAR_DEFAULT_THRESHOLD = 0.20
BLINK_CLOSED_RATIO = 0.60       # threshold = baseline_ear × ratio
BLINK_OPEN_RATIO = 0.75         # hysteresis open threshold
BLINK_SHORT_MAX_MS = 250        # ignore blinks shorter than this
BLINK_LONG_MIN_MS = 800         # intentional blink minimum
BLINK_COOLDOWN_MS = 1000        # cooldown after colour change

# ---------------------------------------------------------------------------
# Colour palette  (8 curated colours)
# ---------------------------------------------------------------------------
PALETTE = [
    (232, 89, 60),
    (83, 74, 183),
    (29, 158, 117),
    (239, 159, 39),
    (212, 83, 126),
    (59, 109, 17),
    (175, 169, 236),
    (245, 245, 240),
]

# ---------------------------------------------------------------------------
# Background & frame rate
# ---------------------------------------------------------------------------
BG_COLOR = (17, 17, 17)
FPS = 60

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(name)-18s] %(levelname)-7s  %(message)s"
LOG_LEVEL = logging.INFO
