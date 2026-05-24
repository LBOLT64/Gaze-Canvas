"""
Gaze Canvas — One Euro Filter for gaze smoothing.

Replaces the naive moving-average with an adaptive low-pass filter
that is smooth when stationary and responsive during fast movement.

Reference: Casiez, Roussel & Vogel, "1€ Filter", CHI 2012.
"""

from __future__ import annotations

import math

from src.config import OEF_MIN_CUTOFF, OEF_BETA, OEF_D_CUTOFF


class _LowPassFilter:
    """Simple first-order exponential low-pass."""

    __slots__ = ("_prev", "_initialised")

    def __init__(self) -> None:
        self._prev: float = 0.0
        self._initialised: bool = False

    def __call__(self, value: float, alpha: float) -> float:
        if not self._initialised:
            self._prev = value
            self._initialised = True
            return value
        filtered = alpha * value + (1.0 - alpha) * self._prev
        self._prev = filtered
        return filtered

    @property
    def last(self) -> float:
        return self._prev

    def reset(self) -> None:
        self._initialised = False


class OneEuroFilter:
    """1€ adaptive low-pass filter for a single scalar signal."""

    def __init__(
        self,
        min_cutoff: float = OEF_MIN_CUTOFF,
        beta: float = OEF_BETA,
        d_cutoff: float = OEF_D_CUTOFF,
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filt = _LowPassFilter()
        self._dx_filt = _LowPassFilter()
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-9))

    def __call__(self, x: float, t: float) -> float:
        """Filter value *x* at time *t* (seconds)."""
        if self._t_prev is None:
            self._t_prev = t
            return self._x_filt(x, 1.0)   # initialise

        dt = t - self._t_prev
        if dt <= 0:
            return self._x_filt.last
        self._t_prev = t

        # Derivative estimate
        dx = (x - self._x_filt.last) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        dx_hat = self._dx_filt(dx, a_d)

        # Adaptive cutoff
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)

        # Filter the signal
        a = self._alpha(cutoff, dt)
        return self._x_filt(x, a)

    def reset(self) -> None:
        self._x_filt.reset()
        self._dx_filt.reset()
        self._t_prev = None


class GazeSmoother:
    """Smooths 2-D gaze coordinates using two independent One Euro Filters."""

    def __init__(
        self,
        min_cutoff: float = OEF_MIN_CUTOFF,
        beta: float = OEF_BETA,
        d_cutoff: float = OEF_D_CUTOFF,
    ) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff

    def update(self, x: float, y: float, t: float) -> tuple[float, float]:
        """Return smoothed (x, y). *t* is in seconds.

        Invalid (NaN / inf) samples are silently rejected.
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            return self._fx._x_filt.last, self._fy._x_filt.last
        return self._fx(x, t), self._fy(y, t)

    def reset(self) -> None:
        self._fx = OneEuroFilter(self._min_cutoff, self._beta, self._d_cutoff)
        self._fy = OneEuroFilter(self._min_cutoff, self._beta, self._d_cutoff)
