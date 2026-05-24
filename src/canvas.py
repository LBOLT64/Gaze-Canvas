"""
Gaze Canvas — drawable surface.

Colour cycling is handled externally via long-blink detection in main.py.
The canvas only draws; it never changes colour on its own.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pygame

from src.config import (
    BRUSH_DEFAULT,
    BRUSH_MAX,
    BRUSH_MIN,
    PALETTE,
)

logger = logging.getLogger(__name__)


class Canvas:
    """RGBA drawing surface that the user paints on with their gaze."""

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.colour_index: int = 0
        self.brush: int = BRUSH_DEFAULT
        self.erase: bool = False

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    def draw_at(self, x: float, y: float, dt_ms: float) -> None:
        """Paint at (*x*, *y*).  *dt_ms* kept for API compat but unused."""
        # Clamp to surface bounds
        ix = max(0, min(int(x), self._width - 1))
        iy = max(0, min(int(y), self._height - 1))

        if self.erase:
            pygame.draw.circle(
                self._surface, (0, 0, 0, 0), (ix, iy), self.brush * 2
            )
        else:
            colour = PALETTE[self.colour_index]
            rgba = (*colour, 180)
            pygame.draw.circle(self._surface, rgba, (ix, iy), self.brush)

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
        logger.info("Canvas cleared")

    def save(self) -> str:
        """Save the canvas to a timestamped PNG and return the path."""
        os.makedirs("saves", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("saves", f"gaze_canvas_{ts}.png")
        pygame.image.save(self._surface, path)
        logger.info("Canvas saved → %s", path)
        return path

    def get_surface(self) -> pygame.Surface:
        return self._surface
