"""
Gaze Canvas — main application entry point.

Run:
    python main.py

The app opens a pygame window, captures your webcam, and tracks your
iris via MediaPipe.  After a 9-point calibration (press SPACE to start
each point's sample collection) your gaze becomes the paintbrush.

Key bindings:
    SPACE   begin calibration sample collection for current point
    Q       quit
    C       recalibrate
    E       toggle eraser
    DEL     clear canvas
    S       save canvas
    [ / ]   decrease / increase brush size
    D       toggle debug overlay
"""

from __future__ import annotations

import logging
import time

import pygame

from src.blink_detector import BlinkDetector
from src.camera import CameraCapture
from src.calibration import CalibrationManager
from src.canvas import Canvas
from src.config import (
    BG_COLOR,
    CAL_POINTS,
    FPS,
    LOG_FORMAT,
    LOG_LEVEL,
    PALETTE,
    SCREEN_H,
    SCREEN_W,
)
from src.gaze_tracker import GazeTracker
from src.smoother import GazeSmoother
from src.ui import UIManager

# ---------------------------------------------------------------------------
# Logging — configured once at the top of the process
# ---------------------------------------------------------------------------
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("gaze_canvas")


class GazeCanvasApp:
    """Top-level application controller."""

    def __init__(self) -> None:
        # ---- Pygame ----------------------------------------------------
        pygame.init()
        self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Gaze Canvas")
        self._clock = pygame.time.Clock()

        # Fonts — fall back to pygame default if the requested family is
        # missing on this system.
        self._fonts = {
            "tiny":   self._safe_font("segoeui", 13),
            "small":  self._safe_font("segoeui", 18),
            "medium": self._safe_font("segoeui", 24),
        }

        # ---- Sub-systems -----------------------------------------------
        self._camera = CameraCapture()
        cam_ok = self._camera.start()

        self._tracker: GazeTracker | None = None
        try:
            self._tracker = GazeTracker()
        except Exception as exc:
            logger.error("Failed to initialise GazeTracker: %s", exc)

        self._smoother = GazeSmoother()           # One Euro Filter (no args)
        self._calibration = CalibrationManager(SCREEN_W, SCREEN_H)
        self._canvas = Canvas(SCREEN_W, SCREEN_H)
        self._blink = BlinkDetector()
        self._ui = UIManager(self._screen, self._fonts)

        # ---- State -----------------------------------------------------
        self._state: str = "calibrating"     # 'calibrating' | 'drawing'
        self._running: bool = True
        self._debug: bool = False
        self._last_frame = None
        self._cal_flash_timer: int = 0

        # Keep the last valid gaze position so the cursor doesn't vanish
        # when tracking is temporarily lost.
        self._last_gaze_screen: tuple[float, float] = (
            SCREEN_W / 2.0, SCREEN_H / 2.0
        )

        # Cached per-frame values for debug overlay
        self._dbg_rel_x: float = 0.5
        self._dbg_rel_y: float = 0.5
        self._dbg_left_ear: float = 0.0
        self._dbg_right_ear: float = 0.0
        self._dbg_avg_ear: float = 0.0
        self._dbg_blink_state: str = "none"
        self._dbg_confidence: bool = False

        # Flag for startup errors to show on-screen
        self._startup_error: str | None = None
        if not cam_ok:
            self._startup_error = "Webcam not found — check connection"
        elif self._tracker is None:
            self._startup_error = "MediaPipe init failed — see console log"

        logger.info("GazeCanvasApp initialised  (state=%s)", self._state)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_font(family: str, size: int) -> pygame.font.Font:
        """Return a SysFont, falling back to the default if *family* is
        unavailable."""
        try:
            return pygame.font.SysFont(family, size)
        except Exception:
            return pygame.font.Font(None, size)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        try:
            while self._running:
                dt_ms = self._clock.tick(FPS)
                t_ms = pygame.time.get_ticks()
                t_sec = time.monotonic()

                self._handle_events()

                # -- camera → tracker -----------------------------------
                frame = self._camera.get_frame()
                if frame is not None:
                    self._last_frame = frame

                gaze = None
                if (
                    self._last_frame is not None
                    and self._tracker is not None
                ):
                    gaze = self._tracker.process(self._last_frame)

                # -- update debug values --------------------------------
                if gaze is not None and gaze.confidence:
                    self._dbg_rel_x = gaze.rel_x
                    self._dbg_rel_y = gaze.rel_y
                    self._dbg_left_ear = gaze.left_ear
                    self._dbg_right_ear = gaze.right_ear
                    self._dbg_avg_ear = gaze.avg_ear
                    self._dbg_confidence = True
                else:
                    self._dbg_confidence = False

                # -- render ---------------------------------------------
                self._screen.fill(BG_COLOR)

                if self._startup_error:
                    self._ui.draw_status_message(
                        self._startup_error, colour=(255, 80, 80)
                    )
                elif self._state == "calibrating":
                    self._draw_calibration(gaze, t_ms)
                else:
                    self._draw_painting(gaze, t_sec, dt_ms, t_ms)

                # -- debug overlay (both states) ------------------------
                if self._debug:
                    fps = self._clock.get_fps()
                    sx, sy = self._last_gaze_screen
                    self._ui.draw_debug(
                        rel_x=self._dbg_rel_x,
                        rel_y=self._dbg_rel_y,
                        screen_x=sx,
                        screen_y=sy,
                        left_ear=self._dbg_left_ear,
                        right_ear=self._dbg_right_ear,
                        avg_ear=self._dbg_avg_ear,
                        blink_state=self._dbg_blink_state,
                        blink_closed=self._blink.is_closed,
                        confidence=self._dbg_confidence,
                        fps=fps,
                        app_state=self._state,
                    )

                pygame.display.flip()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
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
            self._calibration.reset()
            self._smoother.reset()
            self._blink.reset()
            self._state = "calibrating"

        elif key == pygame.K_d:
            self._debug = not self._debug
            logger.info("Debug overlay %s", "ON" if self._debug else "OFF")

        elif key == pygame.K_e:
            self._canvas.set_erase_mode(not self._canvas.erase)

        elif key == pygame.K_DELETE:
            self._canvas.clear()

        elif key == pygame.K_s:
            self._canvas.save()

        elif key == pygame.K_LEFTBRACKET:
            self._canvas.set_brush_size(self._canvas.brush - 2)

        elif key == pygame.K_RIGHTBRACKET:
            self._canvas.set_brush_size(self._canvas.brush + 2)

        elif key == pygame.K_SPACE and self._state == "calibrating":
            self._start_calibration_collection()

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def _start_calibration_collection(self) -> None:
        """User pressed SPACE — begin auto-collecting samples."""
        if self._calibration.is_collecting:
            return                        # already collecting
        self._calibration.start_collecting()

    def _draw_calibration(self, gaze, t_ms: int) -> None:
        tx, ty = self._calibration.current_target()
        idx = self._calibration.current_index
        self._ui.draw_cal_target(tx, ty, idx, CAL_POINTS)

        # --- Auto-collect samples while collecting flag is set ----------
        if self._calibration.is_collecting and gaze is not None and gaze.confidence:
            point_done = self._calibration.feed_sample(
                gaze.rel_x, gaze.rel_y, gaze.avg_ear
            )
            if point_done:
                self._cal_flash_timer = 8

                # If all points collected → switch to drawing
                if self._calibration.is_calibrated():
                    # Finalise blink-detector thresholds from collected EAR
                    for ear in self._calibration.ear_samples:
                        self._blink.feed_calibration_ear(ear)
                    self._blink.finalize_calibration()

                    self._smoother.reset()
                    self._state = "drawing"
                    logger.info("Switched to drawing mode")

        # Show collection progress ring
        if self._calibration.is_collecting:
            self._ui.draw_cal_progress(
                tx, ty, self._calibration.collection_progress
            )

        if self._cal_flash_timer > 0:
            self._ui.draw_cal_flash((100, 200, 255))
            self._cal_flash_timer -= 1

        confidence = gaze is not None and gaze.confidence
        self._ui.draw_confidence(confidence)
        self._ui.draw_webcam_preview(self._last_frame)

    # ------------------------------------------------------------------ #
    # Drawing mode
    # ------------------------------------------------------------------ #

    def _draw_painting(
        self, gaze, t_sec: float, dt_ms: float, t_ms: int
    ) -> None:
        confidence = gaze is not None and gaze.confidence

        if confidence and self._calibration.is_calibrated():
            # --- Smooth the relative gaze, then transform to screen ----
            sx_rel, sy_rel = self._smoother.update(
                gaze.rel_x, gaze.rel_y, t_sec
            )
            sx, sy = self._calibration.transform(sx_rel, sy_rel)
            self._last_gaze_screen = (sx, sy)
            self._canvas.draw_at(sx, sy, dt_ms)

            # --- Blink detection (EAR-based) ---------------------------
            blink_result = self._blink.update(gaze.avg_ear, t_ms)
            self._dbg_blink_state = blink_result
            if blink_result == "long_blink":
                self._canvas.cycle_colour()
                logger.info(
                    "Long blink → colour %d (%s)",
                    self._canvas.colour_index,
                    PALETTE[self._canvas.colour_index],
                )

        # Always blit the canvas
        self._screen.blit(self._canvas.get_surface(), (0, 0))

        # Show the cursor at last known position (stays still when face is
        # lost rather than disappearing)
        gx, gy = self._last_gaze_screen
        self._ui.draw_cursor(gx, gy, t_ms)

        colour = PALETTE[self._canvas.colour_index]
        self._ui.draw_toolbar(self._canvas.brush, colour, self._canvas.erase)
        self._ui.draw_confidence(confidence)
        self._ui.draw_webcam_preview(self._last_frame)

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #

    def _cleanup(self) -> None:
        self._camera.stop()
        if self._tracker is not None:
            self._tracker.close()
        pygame.quit()
        logger.info("Cleanup complete — goodbye")


# ======================================================================= #

if __name__ == "__main__":
    app = GazeCanvasApp()
    app.run()
