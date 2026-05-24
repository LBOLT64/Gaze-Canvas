"""
Gaze Canvas — dual-eye iris tracker with relative positioning and EAR.

Architecture (v2)
-----------------
1. CLAHE lighting normalisation on every frame.
2. Both irises (landmarks 468 & 473) are used.
3. Iris x/y is computed *relative* to the eye-corner bounding box,
   making the signal inherently head-translation compensated.
4. Eye Aspect Ratio (EAR) is computed per-eye for blink detection.
5. All heavy processing is wrapped in try/except so a single
   corrupted frame can never crash the app.

MediaPipe landmark indices used
-------------------------------
Left eye:  outer 33, inner 133, upper 160/158, lower 144/153, iris 468
Right eye: outer 263, inner 362, upper 385/387, lower 373/380, iris 473
Head:      nose tip 1
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Detect which MediaPipe API is available ────────────────────────────
import mediapipe as mp

_HAS_SOLUTIONS = hasattr(mp, "solutions") and hasattr(
    getattr(mp, "solutions", None), "face_mesh"
)
_HAS_TASKS = hasattr(mp, "tasks")

logger.info(
    "mediapipe %s — solutions API: %s, tasks API: %s",
    mp.__version__, _HAS_SOLUTIONS, _HAS_TASKS,
)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")

# ── Landmark indices ───────────────────────────────────────────────────

# Left eye (subject's left — right side of un-mirrored frame)
_L_OUTER = 33        # outer corner
_L_INNER = 133       # inner corner (near nose)
_L_UPPER_1 = 160     # upper lid, outer
_L_UPPER_2 = 158     # upper lid, inner
_L_LOWER_1 = 144     # lower lid, outer
_L_LOWER_2 = 153     # lower lid, inner
_L_IRIS = 468        # iris centre

# Right eye (subject's right — left side of un-mirrored frame)
_R_OUTER = 263       # outer corner
_R_INNER = 362       # inner corner (near nose)
_R_UPPER_1 = 385     # upper lid, inner
_R_UPPER_2 = 387     # upper lid, outer
_R_LOWER_1 = 373     # lower lid, inner
_R_LOWER_2 = 380     # lower lid, outer
_R_IRIS = 473        # iris centre

# Head reference
_NOSE_TIP = 1


# ── Helpers ────────────────────────────────────────────────────────────

def _dist(a, b) -> float:
    """Euclidean distance between two landmarks (normalised coords)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _ear_6(p1, p2, p3, p4, p5, p6) -> float:
    """Eye Aspect Ratio from the canonical 6-landmark configuration.

    EAR = (|P2-P6| + |P3-P5|) / (2 · |P1-P4|)

    P1,P4 = corners (horizontal)
    P2,P3 = upper lid     P5,P6 = lower lid
    """
    vert1 = _dist(p2, p6)
    vert2 = _dist(p3, p5)
    horiz = _dist(p1, p4)
    if horiz < 1e-7:
        return 0.3
    return (vert1 + vert2) / (2.0 * horiz)


# ── Result container ──────────────────────────────────────────────────

@dataclass
class GazeResult:
    """Container returned by ``GazeTracker.process()``."""

    # Relative iris position averaged across both eyes.
    # x: 0 = looking fully right, 1 = looking fully left
    # y: ~0.4 = looking up, ~0.6 = looking down
    rel_x: float = 0.5
    rel_y: float = 0.5

    # Per-eye EAR for blink detection
    left_ear: float = 0.3
    right_ear: float = 0.3
    avg_ear: float = 0.3

    # Was a face (with valid eyes) detected?
    confidence: bool = False

    # Raw MediaPipe face-landmark object for debug drawing
    landmarks: Any = field(default=None, repr=False)

    # Input frame shape (h, w) — for coordinate conversion in debug UI
    frame_shape: tuple[int, int] = (720, 1280)


# ── Main tracker ──────────────────────────────────────────────────────

class GazeTracker:
    """Dual-eye iris tracker with relative positioning and EAR output.

    Automatically selects the legacy ``mp.solutions`` backend (≤ 0.10.20)
    or the modern ``mp.tasks`` backend (≥ 0.10.21).
    """

    def __init__(self) -> None:
        self._backend: str = "none"
        self._face_mesh: Any = None
        self._landmarker: Any = None
        self._frame_index: int = 0

        # Re-use a single CLAHE object for lighting normalisation
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if _HAS_SOLUTIONS:
            self._init_solutions()
        elif _HAS_TASKS:
            self._init_tasks()
        else:
            raise RuntimeError(
                "Neither mediapipe.solutions nor mediapipe.tasks is available. "
                f"Installed version: {mp.__version__}.  "
                "Install a compatible version:  pip install mediapipe==0.10.14"
            )

    # ── Backend initialisers ───────────────────────────────────────────

    def _init_solutions(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,       # iris landmarks 468-477
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._backend = "solutions"
        logger.info("GazeTracker using mp.solutions.face_mesh")

    def _init_tasks(self) -> None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        if not os.path.isfile(_MODEL_PATH):
            os.makedirs(_MODEL_DIR, exist_ok=True)
            logger.info("Downloading face_landmarker.task model …")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            logger.info("Model saved to %s", _MODEL_PATH)

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._backend = "tasks"
        logger.info("GazeTracker using mp.tasks FaceLandmarker (VIDEO mode)")

    # ── Public API ─────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> GazeResult:
        """Run face-mesh on a BGR *frame* and return gaze + EAR data."""
        try:
            return self._process_impl(frame)
        except Exception:
            logger.debug("GazeTracker.process() exception", exc_info=True)
            return GazeResult(confidence=False, frame_shape=frame.shape[:2])

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._landmarker is not None:
            self._landmarker.close()
        logger.info("GazeTracker closed")

    # ── Core processing ────────────────────────────────────────────────

    def _process_impl(self, frame: np.ndarray) -> GazeResult:
        h, w = frame.shape[:2]

        # -- Lighting normalisation via CLAHE on luminance channel ------
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        frame_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # -- Run MediaPipe ----------------------------------------------
        lm = self._run_backend(frame_norm)
        if lm is None:
            return GazeResult(confidence=False, frame_shape=(h, w))

        # -- Extract landmarks as a flat list ---------------------------
        # (both solutions and tasks backends set lm to the landmark list)

        # -- Left eye relative iris position ----------------------------
        l_outer = lm[_L_OUTER]
        l_inner = lm[_L_INNER]
        l_iris  = lm[_L_IRIS]

        l_eye_w = abs(l_outer.x - l_inner.x)
        if l_eye_w < 1e-6:
            return GazeResult(confidence=False, frame_shape=(h, w))

        l_min_x = min(l_outer.x, l_inner.x)
        left_rel_x = (l_iris.x - l_min_x) / l_eye_w

        l_mid_y = (l_outer.y + l_inner.y) / 2.0
        left_rel_y = 0.5 + (l_iris.y - l_mid_y) / l_eye_w

        # -- Right eye relative iris position ---------------------------
        r_outer = lm[_R_OUTER]
        r_inner = lm[_R_INNER]
        r_iris  = lm[_R_IRIS]

        r_eye_w = abs(r_outer.x - r_inner.x)
        if r_eye_w < 1e-6:
            return GazeResult(confidence=False, frame_shape=(h, w))

        r_min_x = min(r_outer.x, r_inner.x)
        right_rel_x = (r_iris.x - r_min_x) / r_eye_w

        r_mid_y = (r_outer.y + r_inner.y) / 2.0
        right_rel_y = 0.5 + (r_iris.y - r_mid_y) / r_eye_w

        # -- Average both eyes -----------------------------------------
        rel_x = (left_rel_x + right_rel_x) / 2.0
        rel_y = (left_rel_y + right_rel_y) / 2.0

        # -- EAR (Eye Aspect Ratio) ------------------------------------
        left_ear = _ear_6(
            lm[_L_OUTER], lm[_L_UPPER_1], lm[_L_UPPER_2],
            lm[_L_INNER], lm[_L_LOWER_2], lm[_L_LOWER_1],
        )
        right_ear = _ear_6(
            lm[_R_INNER], lm[_R_UPPER_1], lm[_R_UPPER_2],
            lm[_R_OUTER], lm[_R_LOWER_2], lm[_R_LOWER_1],
        )
        avg_ear = (left_ear + right_ear) / 2.0

        return GazeResult(
            rel_x=rel_x,
            rel_y=rel_y,
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            confidence=True,
            landmarks=lm,
            frame_shape=(h, w),
        )

    # ── Backend dispatch ───────────────────────────────────────────────

    def _run_backend(self, frame_bgr: np.ndarray):
        """Run the appropriate MediaPipe backend.

        Returns a list-like of landmarks or *None* if no face found.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._backend == "solutions":
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            return results.multi_face_landmarks[0].landmark

        # tasks backend
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_index += 1
        timestamp_ms = self._frame_index * 33
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]
