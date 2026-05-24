"""
Gaze Canvas — iris-based gaze estimation via MediaPipe Face Mesh.

Compatibility
-------------
* **mediapipe ≤ 0.10.20** — uses the legacy ``mp.solutions.face_mesh`` API.
* **mediapipe ≥ 0.10.21** — the ``solutions`` submodule was removed.
  The code falls back to ``mediapipe.tasks.python.vision.FaceLandmarker``,
  which requires a ``.task`` model file that is auto-downloaded on first run.

Both paths expose the same ``GazeTracker.process()`` contract.
"""

from __future__ import annotations

import logging
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

logger.info("mediapipe %s — solutions API: %s, tasks API: %s",
            mp.__version__, _HAS_SOLUTIONS, _HAS_TASKS)

# Model URL for the tasks-API fallback (Google-hosted, ~5 MB)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")

# ── Shared dataclass ───────────────────────────────────────────────────

@dataclass
class GazeResult:
    """Container for a single gaze-estimation result."""
    raw_x: float = 0.0
    raw_y: float = 0.0
    confidence: bool = False
    landmarks: Any = field(default=None, repr=False)


# ── GazeTracker ────────────────────────────────────────────────────────

class GazeTracker:
    """Thin wrapper that auto-selects the best MediaPipe backend.

    Priority:
      1. ``mp.solutions.face_mesh`` (legacy, simpler, no model download)
      2. ``mp.tasks.python.vision.FaceLandmarker`` (modern Tasks API)
    """

    # Iris landmark index (same across both APIs)
    _LEFT_IRIS_CENTER = 468

    def __init__(self) -> None:
        self._backend: str = "none"
        self._face_mesh: Any = None        # legacy
        self._landmarker: Any = None       # tasks
        self._frame_index: int = 0         # tasks-VIDEO needs a timestamp

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

    # -- legacy mp.solutions initialiser ---------------------------------
    def _init_solutions(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,           # enables iris 468-477
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._backend = "solutions"
        logger.info("GazeTracker using legacy mp.solutions.face_mesh")

    # -- modern mp.tasks initialiser -------------------------------------
    def _init_tasks(self) -> None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        # Auto-download the model if missing
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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process(self, frame: np.ndarray) -> GazeResult:
        """Run face-mesh on a BGR *frame* and return normalised iris coords."""
        try:
            if self._backend == "solutions":
                return self._process_solutions(frame)
            else:
                return self._process_tasks(frame)
        except Exception:
            logger.debug("GazeTracker.process() exception", exc_info=True)
            return GazeResult(confidence=False)

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._landmarker is not None:
            self._landmarker.close()
        logger.info("GazeTracker closed")

    # ------------------------------------------------------------------ #
    # Backend implementations
    # ------------------------------------------------------------------ #

    def _process_solutions(self, frame: np.ndarray) -> GazeResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return GazeResult(confidence=False)

        landmarks = results.multi_face_landmarks[0]
        iris = landmarks.landmark[self._LEFT_IRIS_CENTER]
        return GazeResult(
            raw_x=iris.x,
            raw_y=iris.y,
            confidence=True,
            landmarks=landmarks,
        )

    def _process_tasks(self, frame: np.ndarray) -> GazeResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_index += 1
        # Tasks VIDEO mode requires monotonically-increasing timestamp (ms)
        timestamp_ms = self._frame_index * 33   # ~30 fps spacing

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return GazeResult(confidence=False)

        landmarks = result.face_landmarks[0]
        iris = landmarks[self._LEFT_IRIS_CENTER]
        return GazeResult(
            raw_x=iris.x,
            raw_y=iris.y,
            confidence=True,
            landmarks=landmarks,
        )
