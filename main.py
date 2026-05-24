"""
Gaze Canvas — main application entry point.

Run:
    python main.py
"""

from __future__ import annotations

import sys

import pygame

from src.camera import CameraCapture
from src.calibration import CalibrationManager
from src.canvas import Canvas
from src.config import (
    BG_COLOR,
    BRUSH_MAX,
    BRUSH_MIN,
    CAL_POINTS,
    FPS,
    PALETTE,
    SCREEN_H,
    SCREEN_W,
    SMOOTHING_WINDOW,
)
from src.gaze_tracker import GazeTracker
from src.smoother import GazeSmoother
from src.ui import UIManager


class GazeCanvasApp:
    """Top-level application controller."""

    def __init__(self) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Gaze Canvas")
        self._clock = pygame.time.Clock()

        # Fonts
        self._fonts = {
            "tiny": pygame.font.SysFont("segoeui", 13),
            "small": pygame.font.SysFont("segoeui", 18),
            "medium": pygame.font.SysFont("segoeui", 24),
        }

        # Sub-systems
        self._camera = CameraCapture()
        self._tracker = GazeTracker()
        self._smoother = GazeSmoother(window=SMOOTHING_WINDOW)
        self._calibration = CalibrationManager(SCREEN_W, SCREEN_H)
        self._canvas = Canvas(SCREEN_W, SCREEN_H)
        self._ui = UIManager(self._screen, self._fonts)

        # State
        self._state: str = "calibrating"  # 'calibrating' | 'drawing'
        self._running: bool = True
        self._last_frame = None
        self._cal_flash_timer: int = 0

        self._camera.start()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        while self._running:
            dt_ms = self._clock.tick(FPS)
            t_ms = pygame.time.get_ticks()

            self._handle_events()

            # --- camera → tracker → smoother ---
            frame = self._camera.get_frame()
            if frame is not None:
                self._last_frame = frame

            gaze = None
            smoothed = None
            if self._last_frame is not None:
                gaze = self._tracker.process(self._last_frame)
                if gaze.confidence:
                    smoothed = self._smoother.update(gaze.raw_x, gaze.raw_y)

            # --- render ---
            self._screen.fill(BG_COLOR)

            if self._state == "calibrating":
                self._draw_calibration(gaze, t_ms)
            else:
                self._draw_painting(gaze, smoothed, dt_ms, t_ms)

            pygame.display.flip()

        self._cleanup()

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                self._on_key(event.key)

    def _on_key(self, key: int) -> None:
        if key == pygame.K_q:
            self._running = False

        elif key == pygame.K_c:
            # Recalibrate
            self._calibration.reset()
            self._smoother.reset()
            self._state = "calibrating"

        elif key == pygame.K_e:
            self._canvas.set_erase_mode(not self._canvas.erase)

        elif key == pygame.K_DELETE:
            self._canvas.clear()

        elif key == pygame.K_s:
            path = self._canvas.save()
            print(f"[Gaze Canvas] Saved → {path}")

        elif key == pygame.K_LEFTBRACKET:
            self._canvas.set_brush_size(self._canvas.brush - 2)

        elif key == pygame.K_RIGHTBRACKET:
            self._canvas.set_brush_size(self._canvas.brush + 2)

        elif key == pygame.K_SPACE and self._state == "calibrating":
            self._record_calibration_sample()

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def _record_calibration_sample(self) -> None:
        """Record the current smoothed gaze for the active cal target."""
        if self._last_frame is None:
            return
        gaze = self._tracker.process(self._last_frame)
        if not gaze.confidence:
            return

        done = self._calibration.record_sample(gaze.raw_x, gaze.raw_y)
        self._cal_flash_timer = 8  # flash for N frames

        if done:
            self._state = "drawing"
            self._smoother.reset()

    def _draw_calibration(self, gaze, t_ms: int) -> None:
        tx, ty = self._calibration.current_target()
        idx = self._calibration._index  # noqa: SLF001
        self._ui.draw_cal_target(tx, ty, idx, CAL_POINTS)

        if self._cal_flash_timer > 0:
            self._ui.draw_cal_flash((100, 200, 255))
            self._cal_flash_timer -= 1

        if gaze and gaze.confidence:
            self._ui.draw_confidence(True)
        else:
            self._ui.draw_confidence(False)

        if self._last_frame is not None:
            self._ui.draw_webcam_preview(self._last_frame)

    # ------------------------------------------------------------------ #
    # Drawing mode
    # ------------------------------------------------------------------ #

    def _draw_painting(self, gaze, smoothed, dt_ms: float, t_ms: int) -> None:
        confidence = gaze is not None and gaze.confidence

        if confidence and smoothed is not None and self._calibration.is_calibrated():
            sx, sy = self._calibration.transform(*smoothed)
            self._canvas.draw_at(sx, sy, dt_ms)
            # Blit canvas then cursor
            self._screen.blit(self._canvas.get_surface(), (0, 0))
            self._ui.draw_cursor(sx, sy, t_ms)
        else:
            self._screen.blit(self._canvas.get_surface(), (0, 0))

        colour = PALETTE[self._canvas.colour_index]
        self._ui.draw_toolbar(self._canvas.brush, colour, self._canvas.erase)
        self._ui.draw_confidence(confidence)

        if self._last_frame is not None:
            self._ui.draw_webcam_preview(self._last_frame)

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #

    def _cleanup(self) -> None:
        self._camera.stop()
        self._tracker.close()
        pygame.quit()


if __name__ == "__main__":
    app = GazeCanvasApp()
    app.run()
