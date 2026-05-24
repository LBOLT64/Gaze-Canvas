"""
Gaze Canvas — iris-based gaze estimation via MediaPipe Face Mesh.
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class GazeResult:
    """Container for a single gaze-estimation result."""
    raw_x: float = 0.0
    raw_y: float = 0.0
    confidence: bool = False
    landmarks: Any = field(default=None, repr=False)


class GazeTracker:
    """Thin wrapper around MediaPipe Face Mesh with iris refinement."""

    # Iris landmark indices added by refine_landmarks
    _LEFT_IRIS_CENTER = 468

    def __init__(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame: np.ndarray) -> GazeResult:
        """Run face-mesh on a BGR frame and return normalised iris coords."""
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

    def close(self) -> None:
        self._face_mesh.close()
