"""
Gaze Canvas — drawable surface with dwell-based colour cycling.
"""

from __future__ import annotations

import math
import os
from datetime import datetime

import pygame

from src.config import (
    BRUSH_DEFAULT,
    BRUSH_MAX,
    BRUSH_MIN,
    DWELL_RADIUS_PX,
    DWELL_TIME_MS,
    PALETTE,
)


class Canvas:
    """RGBA drawing surface that the user paints on with their gaze."""

    def __init__(self, width: int, height: int) -> None:
        self._surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.colour_index: int = 0
        self.brush: int = BRUSH_DEFAULT
        self.erase: bool = False

        # Dwell tracking
        self._dwell_start: int | None = None
        self._last_pos: tuple[float, float] | None = None

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    def draw_at(self, x: float, y: float, dt_ms: float) -> bool:
        """
        Paint at (*x*, *y*).

        Returns *True* if a dwell event fired (colour was cycled).
        """
        ix, iy = int(x), int(y)

        if self.erase:
            pygame.draw.circle(
                self._surface, (0, 0, 0, 0), (ix, iy), self.brush * 2
            )
        else:
            colour = PALETTE[self.colour_index]
            rgba = (*colour, 180)
            pygame.draw.circle(self._surface, rgba, (ix, iy), self.brush)

        # --- dwell detection ---
        dwelled = False
        if self._last_pos is not None:
            dist = math.hypot(x - self._last_pos[0], y - self._last_pos[1])
            if dist < DWELL_RADIUS_PX:
                if self._dwell_start is None:
                    self._dwell_start = 0
                self._dwell_start += dt_ms
                if self._dwell_start >= DWELL_TIME_MS:
                    self.cycle_colour()
                    self._dwell_start = None
                    dwelled = True
            else:
                self._dwell_start = None
        self._last_pos = (x, y)
        return dwelled

    # ------------------------------------------------------------------ #
    # Brush / colour helpers
    # ------------------------------------------------------------------ #

    def set_brush_size(self, px: int) -> None:
        self.brush = max(BRUSH_MIN, min(BRUSH_MAX, px))

    def set_erase_mode(self, on: bool) -> None:
        self.erase = on

    def cycle_colour(self) -> None:
        self.colour_index = (self.colour_index + 1) % len(PALETTE)

    # ------------------------------------------------------------------ #
    # Surface management
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        self._surface.fill((0, 0, 0, 0))

    def save(self) -> str:
        """Save the canvas to a timestamped PNG and return the path."""
        os.makedirs("saves", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("saves", f"gaze_canvas_{ts}.png")
        pygame.image.save(self._surface, path)
        return path

    def get_surface(self) -> pygame.Surface:
        return self._surface
