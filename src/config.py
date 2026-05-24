"""
Gaze Canvas — module-level configuration constants.
"""

# Display
SCREEN_W, SCREEN_H = 1280, 800

# Gaze smoothing
SMOOTHING_WINDOW = 10

# Dwell detection
DWELL_RADIUS_PX = 35
DWELL_TIME_MS = 1500

# Brush
BRUSH_MIN, BRUSH_MAX = 4, 36
BRUSH_DEFAULT = 14

# Calibration
CAL_POINTS = 9

# Colour palette
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

# Background
BG_COLOR = (17, 17, 17)

# Frame rate
FPS = 60
