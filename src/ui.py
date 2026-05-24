"""
Gaze Canvas — HUD overlays: cursor, toolbar, confidence dot, webcam pip.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pygame

from src.config import PALETTE


class UIManager:
    """Draws all non-canvas visual elements (toolbar, cursor, previews)."""

    def __init__(self, screen: pygame.Surface, fonts: dict) -> None:
        self._screen = screen
        self._fonts = fonts

    # ------------------------------------------------------------------ #
    # Cursor
    # ------------------------------------------------------------------ #

    def draw_cursor(self, x: float, y: float, t_ms: int) -> None:
        """Pulsing ring whose radius oscillates between 12 and 18 px."""
        radius = int(15 + 3 * math.sin(t_ms / 300.0))
        pygame.draw.circle(
            self._screen, (255, 255, 255), (int(x), int(y)), radius, 2
        )

    # ------------------------------------------------------------------ #
    # Toolbar
    # ------------------------------------------------------------------ #

    def draw_toolbar(
        self, brush_size: int, colour: tuple, erase_mode: bool
    ) -> None:
        """40 px bar along the bottom with colour swatch + info."""
        sw, sh = self._screen.get_size()
        bar_rect = pygame.Rect(0, sh - 40, sw, 40)
        bar_surf = pygame.Surface((sw, 40), pygame.SRCALPHA)
        bar_surf.fill((30, 30, 30, 200))
        self._screen.blit(bar_surf, bar_rect.topleft)

        # Colour swatch
        pygame.draw.circle(self._screen, colour, (30, sh - 20), 12)
        pygame.draw.circle(self._screen, (200, 200, 200), (30, sh - 20), 12, 1)

        # Brush size label
        label = self._fonts["small"].render(
            f"Brush: {brush_size}px", True, (220, 220, 220)
        )
        self._screen.blit(label, (55, sh - 30))

        # Erase indicator
        if erase_mode:
            tag = self._fonts["small"].render("ERASER", True, (255, 80, 80))
            self._screen.blit(tag, (180, sh - 30))

        # Keyboard hints
        hints = self._fonts["tiny"].render(
            "Q quit · C calibrate · E erase · DEL clear · S save · [ ] brush",
            True,
            (130, 130, 130),
        )
        self._screen.blit(hints, (sw // 2 - hints.get_width() // 2, sh - 18))

    # ------------------------------------------------------------------ #
    # Confidence indicator
    # ------------------------------------------------------------------ #

    def draw_confidence(self, ok: bool) -> None:
        """Small coloured dot in the top-left corner."""
        colour = (0, 200, 80) if ok else (220, 40, 40)
        pygame.draw.circle(self._screen, colour, (16, 16), 8)

    # ------------------------------------------------------------------ #
    # Webcam PiP
    # ------------------------------------------------------------------ #

    def draw_webcam_preview(self, frame: np.ndarray) -> None:
        """Scale the webcam frame to 160×90 and draw in the bottom-right."""
        small = cv2.resize(frame, (160, 90))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.transpose(small, (1, 0, 2)))
        sw, sh = self._screen.get_size()
        self._screen.blit(surf, (sw - 170, sh - 50 - 90))

    # ------------------------------------------------------------------ #
    # Calibration visuals
    # ------------------------------------------------------------------ #

    def draw_cal_target(
        self, x: int, y: int, index: int, total: int
    ) -> None:
        """White filled circle + progress label."""
        pygame.draw.circle(self._screen, (255, 255, 255), (x, y), 18)
        pygame.draw.circle(self._screen, (100, 100, 100), (x, y), 18, 2)

        label = self._fonts["small"].render(
            f"{index + 1}/{total}", True, (200, 200, 200)
        )
        self._screen.blit(label, (x - label.get_width() // 2, y + 28))

        # Instruction text
        instr = self._fonts["medium"].render(
            "Look at the dot and press SPACE", True, (180, 180, 180)
        )
        sw, _ = self._screen.get_size()
        self._screen.blit(instr, (sw // 2 - instr.get_width() // 2, 30))

    def draw_cal_flash(self, colour: tuple) -> None:
        """Brief full-screen tinted overlay."""
        sw, sh = self._screen.get_size()
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((*colour, 50))
        self._screen.blit(overlay, (0, 0))
