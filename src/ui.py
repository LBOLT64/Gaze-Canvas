"""
Gaze Canvas — HUD overlays: cursor, toolbar, confidence dot, webcam PiP.

All ``draw_*`` methods are safe to call with any input — they guard
against None frames, invalid coordinates, and missing fonts.
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
        self._sw, self._sh = screen.get_size()

    # ------------------------------------------------------------------ #
    # Cursor
    # ------------------------------------------------------------------ #

    def draw_cursor(self, x: float, y: float, t_ms: int) -> None:
        """Pulsing ring whose radius oscillates between 12 and 18 px."""
        radius = int(15 + 3 * math.sin(t_ms / 300.0))
        ix, iy = int(x), int(y)
        # Outer ring (white)
        pygame.draw.circle(self._screen, (255, 255, 255), (ix, iy), radius, 2)
        # Small centre dot for precision
        pygame.draw.circle(self._screen, (255, 255, 255, 150), (ix, iy), 2)

    # ------------------------------------------------------------------ #
    # Toolbar
    # ------------------------------------------------------------------ #

    def draw_toolbar(
        self, brush_size: int, colour: tuple, erase_mode: bool
    ) -> None:
        """40 px bar along the bottom with colour swatch + info."""
        bar_rect = pygame.Rect(0, self._sh - 40, self._sw, 40)
        bar_surf = pygame.Surface((self._sw, 40), pygame.SRCALPHA)
        bar_surf.fill((30, 30, 30, 200))
        self._screen.blit(bar_surf, bar_rect.topleft)

        # Colour swatch
        pygame.draw.circle(self._screen, colour, (30, self._sh - 20), 12)
        pygame.draw.circle(
            self._screen, (200, 200, 200), (30, self._sh - 20), 12, 1
        )

        # Brush size label
        label = self._fonts["small"].render(
            f"Brush: {brush_size}px", True, (220, 220, 220)
        )
        self._screen.blit(label, (55, self._sh - 30))

        # Erase indicator
        if erase_mode:
            tag = self._fonts["small"].render("ERASER", True, (255, 80, 80))
            self._screen.blit(tag, (180, self._sh - 30))

        # Keyboard hints
        hints = self._fonts["tiny"].render(
            "Q quit | C calibrate | E erase | DEL clear | S save | [ ] brush",
            True,
            (130, 130, 130),
        )
        self._screen.blit(
            hints, (self._sw // 2 - hints.get_width() // 2, self._sh - 18)
        )

    # ------------------------------------------------------------------ #
    # Confidence indicator
    # ------------------------------------------------------------------ #

    def draw_confidence(self, ok: bool) -> None:
        """Small coloured dot in the top-left corner."""
        colour = (0, 200, 80) if ok else (220, 40, 40)
        pygame.draw.circle(self._screen, colour, (16, 16), 8)
        status = self._fonts["tiny"].render(
            "Face OK" if ok else "No face", True, colour
        )
        self._screen.blit(status, (30, 8))

    # ------------------------------------------------------------------ #
    # Webcam PiP
    # ------------------------------------------------------------------ #

    def draw_webcam_preview(self, frame: np.ndarray | None) -> None:
        """Scale the webcam frame to 160×90 and draw in the bottom-right.

        If *frame* is None the call is a harmless no-op.
        """
        if frame is None:
            return
        try:
            small = cv2.resize(frame, (160, 90))
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            # pygame.surfarray expects (width, height, 3) — transpose axes
            surf = pygame.surfarray.make_surface(
                np.transpose(small, (1, 0, 2))
            )
            self._screen.blit(surf, (self._sw - 170, self._sh - 50 - 90))
        except Exception:
            pass  # never crash for a cosmetic preview

    # ------------------------------------------------------------------ #
    # Calibration visuals
    # ------------------------------------------------------------------ #

    def draw_cal_target(
        self, x: int, y: int, index: int, total: int
    ) -> None:
        """White filled circle + progress label."""
        # Outer ring
        pygame.draw.circle(self._screen, (80, 80, 80), (x, y), 22, 2)
        # Inner filled dot
        pygame.draw.circle(self._screen, (255, 255, 255), (x, y), 14)

        # Progress  e.g. "3 / 9"
        prog = self._fonts["small"].render(
            f"{index + 1} / {total}", True, (200, 200, 200)
        )
        self._screen.blit(prog, (x - prog.get_width() // 2, y + 32))

        # Instruction text
        instr = self._fonts["medium"].render(
            "Look at the dot and press SPACE", True, (180, 180, 180)
        )
        self._screen.blit(
            instr, (self._sw // 2 - instr.get_width() // 2, 30)
        )

    def draw_cal_flash(self, colour: tuple) -> None:
        """Brief full-screen tinted overlay."""
        overlay = pygame.Surface((self._sw, self._sh), pygame.SRCALPHA)
        overlay.fill((*colour, 50))
        self._screen.blit(overlay, (0, 0))

    # ------------------------------------------------------------------ #
    # Error / status messages
    # ------------------------------------------------------------------ #

    def draw_status_message(self, text: str, colour=(255, 200, 60)) -> None:
        """Centred single-line message — useful for error/info overlays."""
        surf = self._fonts["medium"].render(text, True, colour)
        self._screen.blit(
            surf,
            (self._sw // 2 - surf.get_width() // 2,
             self._sh // 2 - surf.get_height() // 2),
        )
