"""
Gaze Canvas — EAR-based blink detector with adaptive thresholds.

Key design decisions
--------------------
* Uses Eye Aspect Ratio (EAR) — NOT iris y-position — so looking
  downward is never confused with a blink.
* Hysteresis between closed/open thresholds prevents flutter.
* Only intentional long blinks (> 800 ms) trigger colour changes.
* 1-second cooldown prevents accidental rapid cycling.
* Baseline EAR is learned during calibration for per-user adaptation.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from src.config import (
    BLINK_CLOSED_RATIO,
    BLINK_COOLDOWN_MS,
    BLINK_EAR_DEFAULT_THRESHOLD,
    BLINK_LONG_MIN_MS,
    BLINK_OPEN_RATIO,
    BLINK_SHORT_MAX_MS,
)

logger = logging.getLogger(__name__)


class BlinkDetector:
    """Detects intentional long blinks from an EAR time-series."""

    def __init__(self) -> None:
        # Adaptive threshold state
        self._baseline_ear: float | None = None
        self._closed_thresh: float = BLINK_EAR_DEFAULT_THRESHOLD
        self._open_thresh: float = BLINK_EAR_DEFAULT_THRESHOLD * 1.15

        # Blink state machine
        self._is_closed: bool = False
        self._close_start_ms: int = 0
        self._last_trigger_ms: int = -BLINK_COOLDOWN_MS  # allow immediate first blink

        # Calibration EAR samples (collected during calibration phase)
        self._cal_ears: list[float] = []

        # Recent EAR history for smoothing
        self._ear_buf: deque[float] = deque(maxlen=5)

    # ------------------------------------------------------------------ #
    # Calibration integration
    # ------------------------------------------------------------------ #

    def feed_calibration_ear(self, ear: float) -> None:
        """Collect an EAR sample during calibration for baseline computation."""
        if ear > 0.05:                       # discard closed-eye outliers
            self._cal_ears.append(ear)

    def finalize_calibration(self) -> None:
        """Compute personalised thresholds from calibration EAR data."""
        if len(self._cal_ears) < 20:
            logger.warning(
                "Not enough EAR calibration data (%d samples) — using defaults",
                len(self._cal_ears),
            )
            return

        self._baseline_ear = float(np.mean(self._cal_ears))
        self._closed_thresh = self._baseline_ear * BLINK_CLOSED_RATIO
        self._open_thresh = self._baseline_ear * BLINK_OPEN_RATIO
        logger.info(
            "Blink thresholds calibrated — baseline EAR=%.3f  "
            "closed<%.3f  open>%.3f",
            self._baseline_ear,
            self._closed_thresh,
            self._open_thresh,
        )

    @property
    def baseline_ear(self) -> float | None:
        return self._baseline_ear

    # ------------------------------------------------------------------ #
    # Per-frame update
    # ------------------------------------------------------------------ #

    def update(self, avg_ear: float, t_ms: int) -> str:
        """Feed the current average EAR and timestamp.

        Returns
        -------
        ``'long_blink'`` if an intentional long blink just ended,
        ``'none'`` otherwise.
        """
        # Smooth the EAR signal slightly to reject single-frame noise
        self._ear_buf.append(avg_ear)
        smoothed_ear = float(np.mean(self._ear_buf))

        # --- State machine: OPEN → CLOSED → OPEN ---
        if not self._is_closed:
            if smoothed_ear < self._closed_thresh:
                self._is_closed = True
                self._close_start_ms = t_ms
        else:
            if smoothed_ear > self._open_thresh:
                # Eyes just re-opened
                self._is_closed = False
                duration = t_ms - self._close_start_ms

                # Ignore very short blinks (natural reflex)
                if duration < BLINK_SHORT_MAX_MS:
                    return "none"

                # Check cooldown
                if t_ms - self._last_trigger_ms < BLINK_COOLDOWN_MS:
                    return "none"

                # Long intentional blink
                if duration >= BLINK_LONG_MIN_MS:
                    self._last_trigger_ms = t_ms
                    logger.info(
                        "Long blink detected (%d ms)", duration
                    )
                    return "long_blink"

        return "none"

    # ------------------------------------------------------------------ #
    # State queries
    # ------------------------------------------------------------------ #

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def smoothed_ear(self) -> float:
        return float(np.mean(self._ear_buf)) if self._ear_buf else 0.3

    def reset(self) -> None:
        """Full reset — thresholds revert to defaults."""
        self._baseline_ear = None
        self._closed_thresh = BLINK_EAR_DEFAULT_THRESHOLD
        self._open_thresh = BLINK_EAR_DEFAULT_THRESHOLD * 1.15
        self._is_closed = False
        self._close_start_ms = 0
        self._last_trigger_ms = -BLINK_COOLDOWN_MS
        self._cal_ears.clear()
        self._ear_buf.clear()
