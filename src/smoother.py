"""
Gaze Canvas — moving-average smoother for raw gaze coordinates.
"""

from collections import deque

import numpy as np


class GazeSmoother:
    """Rolling-window mean filter over (x, y) gaze samples."""

    def __init__(self, window: int = 10) -> None:
        self._buf: deque[tuple[float, float]] = deque(maxlen=window)

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Append a sample and return the smoothed (x, y) mean."""
        self._buf.append((x, y))
        mean = np.mean(self._buf, axis=0)
        return float(mean[0]), float(mean[1])

    def reset(self) -> None:
        self._buf.clear()

    def is_ready(self) -> bool:
        """True once the buffer is completely filled."""
        return len(self._buf) == self._buf.maxlen
