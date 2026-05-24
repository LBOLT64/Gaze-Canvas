"""
Gaze Canvas — 9-point calibration with multi-sample collection and
PolynomialFeatures + Ridge regression.

Improvements over v1
--------------------
* Collects 25 samples per calibration point (not 1).
* Rejects outlier samples beyond 2σ.
* Uses degree-2 polynomial features for non-linear screen mapping.
* Collects EAR samples during calibration for blink-threshold tuning.
* ``transform()`` always returns valid, clamped screen coordinates.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.config import CAL_OUTLIER_SIGMA, CAL_SAMPLES_PER_POINT

logger = logging.getLogger(__name__)


class CalibrationManager:
    """Collects gaze→screen samples and fits a polynomial Ridge model."""

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
        self.model: Pipeline | None = None

        # --- Multi-sample state ---
        self._collecting: bool = False
        self._current_samples: list[list[float]] = []
        self._ear_samples: list[float] = []     # for blink-detector calibration

    # ------------------------------------------------------------------ #
    # Public queries
    # ------------------------------------------------------------------ #

    def is_calibrated(self) -> bool:
        return self.model is not None

    @property
    def current_index(self) -> int:
        """How many calibration points have been fully recorded."""
        return self._index

    @property
    def is_collecting(self) -> bool:
        """True while we are auto-collecting samples for a point."""
        return self._collecting

    @property
    def collection_progress(self) -> float:
        """0.0 → 1.0 progress of the current sample collection."""
        if not self._collecting:
            return 0.0
        return len(self._current_samples) / CAL_SAMPLES_PER_POINT

    def current_target(self) -> tuple[int, int]:
        """Screen position of the calibration dot the user should look at."""
        if self._index >= len(self._targets):
            return self._targets[-1]
        return self._targets[self._index]

    @property
    def ear_samples(self) -> list[float]:
        """EAR values collected during calibration (for blink-detector)."""
        return self._ear_samples

    # ------------------------------------------------------------------ #
    # Sample collection
    # ------------------------------------------------------------------ #

    def start_collecting(self) -> None:
        """Call when the user presses SPACE — begins auto-collection."""
        if self._index >= len(self._targets):
            return
        self._collecting = True
        self._current_samples.clear()
        logger.info(
            "Collecting samples for cal point %d/%d …",
            self._index + 1, len(self._targets),
        )

    def feed_sample(self, rel_x: float, rel_y: float,
                    avg_ear: float | None = None) -> bool:
        """Feed one gaze frame during collection.

        Returns *True* when this calibration point is done.
        """
        if not self._collecting:
            return False

        self._current_samples.append([rel_x, rel_y])
        if avg_ear is not None and avg_ear > 0.05:
            self._ear_samples.append(avg_ear)

        if len(self._current_samples) >= CAL_SAMPLES_PER_POINT:
            self._finalize_point()
            return True
        return False

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _finalize_point(self) -> None:
        """Average collected samples, reject outliers, store the result."""
        samples = np.array(self._current_samples)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)

        # Reject outliers — keep only samples within σ threshold
        if std.max() > 1e-8:
            dists = np.abs(samples - mean)
            mask = np.all(dists < CAL_OUTLIER_SIGMA * (std + 1e-9), axis=1)
            clean = samples[mask]
            if len(clean) < 5:
                clean = samples        # fallback if too many rejected
        else:
            clean = samples

        final = clean.mean(axis=0).tolist()
        tx, ty = self._targets[self._index]
        self._X_train.append(final)
        self._y_train.append([tx, ty])
        self._index += 1
        self._collecting = False

        logger.info(
            "Cal point %d/%d done — kept %d/%d samples — "
            "mean (%.4f, %.4f) → (%d, %d)",
            self._index, len(self._targets),
            len(clean), len(self._current_samples),
            final[0], final[1], tx, ty,
        )

        if self._index >= len(self._targets):
            self._fit_model()

    def _fit_model(self) -> None:
        """Fit PolynomialFeatures(degree=2) + Ridge on collected data."""
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=True)),
            ("ridge", Ridge(alpha=1.0)),
        ])
        self.model.fit(
            np.array(self._X_train),
            np.array(self._y_train),
        )
        logger.info("Calibration complete — polynomial Ridge model fitted")

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #

    def transform(self, rel_x: float, rel_y: float) -> tuple[float, float]:
        """Map relative gaze to screen coords via the fitted model.

        Returns screen-centre if the model is not yet available.
        """
        if self.model is None:
            return self._sw / 2.0, self._sh / 2.0
        try:
            pred = self.model.predict(np.array([[rel_x, rel_y]]))[0]
            sx = float(np.clip(pred[0], 0, self._sw))
            sy = float(np.clip(pred[1], 0, self._sh))
            return sx, sy
        except Exception:
            logger.debug("calibration.transform() failed", exc_info=True)
            return self._sw / 2.0, self._sh / 2.0

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear all training data and the fitted model."""
        self._X_train.clear()
        self._y_train.clear()
        self._index = 0
        self.model = None
        self._collecting = False
        self._current_samples.clear()
        self._ear_samples.clear()
        logger.info("Calibration reset")
