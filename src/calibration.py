"""
Gaze Canvas — 9-point calibration with Ridge regression.

Safety: ``transform()`` always returns valid screen coordinates even if
the model is not yet fitted (returns screen-centre as fallback).
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


class CalibrationManager:
    """Collects gaze→screen samples and fits a Ridge regressor."""

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self._sw = screen_w
        self._sh = screen_h

        # Build a 3×3 grid with 20 % margin on each side
        margin_x = int(screen_w * 0.20)
        margin_y = int(screen_h * 0.20)
        xs = np.linspace(margin_x, screen_w - margin_x, 3, dtype=int)
        ys = np.linspace(margin_y, screen_h - margin_y, 3, dtype=int)
        self._targets: list[tuple[int, int]] = [
            (int(x), int(y)) for y in ys for x in xs
        ]

        self._X_train: list[list[float]] = []
        self._y_train: list[list[int]] = []
        self._index: int = 0
        self.model: Ridge | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def is_calibrated(self) -> bool:
        return self.model is not None

    @property
    def current_index(self) -> int:
        """How many calibration points have been recorded so far."""
        return self._index

    def current_target(self) -> tuple[int, int]:
        """Screen position of the calibration dot the user should look at."""
        if self._index >= len(self._targets):
            return self._targets[-1]
        return self._targets[self._index]

    def record_sample(self, raw_x: float, raw_y: float) -> bool:
        """
        Record one gaze sample for the current target.

        Returns *True* when all 9 points have been collected and the
        model is fitted.
        """
        if self._index >= len(self._targets):
            return True                       # already calibrated

        tx, ty = self._targets[self._index]
        self._X_train.append([raw_x, raw_y])
        self._y_train.append([tx, ty])
        self._index += 1

        logger.info("Calibration sample %d/%d recorded  (raw %.4f, %.4f) → (%d, %d)",
                     self._index, len(self._targets), raw_x, raw_y, tx, ty)

        if self._index >= len(self._targets):
            self.model = Ridge(alpha=1.0)
            self.model.fit(
                np.array(self._X_train),
                np.array(self._y_train),
            )
            logger.info("Calibration complete — Ridge model fitted")
            return True
        return False

    def transform(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """Map normalised gaze to screen coords via the fitted model.

        Returns screen-centre if the model is not yet available (safety).
        """
        if self.model is None:
            return self._sw / 2.0, self._sh / 2.0
        try:
            pred = self.model.predict(np.array([[raw_x, raw_y]]))[0]
            sx = float(np.clip(pred[0], 0, self._sw))
            sy = float(np.clip(pred[1], 0, self._sh))
            return sx, sy
        except Exception:
            logger.debug("calibration.transform() failed", exc_info=True)
            return self._sw / 2.0, self._sh / 2.0

    def reset(self) -> None:
        """Clear all training data and the fitted model."""
        self._X_train.clear()
        self._y_train.clear()
        self._index = 0
        self.model = None
        logger.info("Calibration reset")
