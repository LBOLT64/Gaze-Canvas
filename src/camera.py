"""
Gaze Canvas — threaded webcam capture with a bounded frame queue.

Improvements over v1
--------------------
* Forces 1280×720 resolution for consistent landmark quality.
* Uses DirectShow backend on Windows for lower latency.
* Minimises internal OpenCV buffer to 1 frame.
* Thread-safe shutdown via lock + event.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading

import cv2
import numpy as np

from src.config import CAM_WIDTH, CAM_HEIGHT

logger = logging.getLogger(__name__)


class CameraCapture:
    """Continuously grabs webcam frames on a background daemon thread."""

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id
        self._cap: cv2.VideoCapture | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> bool:
        """Open the camera and start the reader thread.

        Returns True if the webcam opened successfully.
        """
        # Use DirectShow on Windows for lower latency
        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(self._device_id, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self._device_id)

        if not self._cap.isOpened():
            logger.error("Could not open webcam (device %s)", self._device_id)
            return False

        # Force resolution and minimise internal buffer
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Webcam opened — %.0fx%.0f @ %.0f fps",
            actual_w, actual_h, actual_fps,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return True

    def get_frame(self) -> np.ndarray | None:
        """Return the latest frame without blocking, or *None*."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Signal the thread to stop and release the camera."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                self._cap.release()
                logger.info("Webcam released")

    def is_open(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _reader(self) -> None:
        """Loop that fills the queue, dropping stale frames when full."""
        while not self._stop_event.is_set():
            with self._lock:
                if self._cap is None or not self._cap.isOpened():
                    logger.warning("Webcam disconnected — reader stopping")
                    break
                ok, frame = self._cap.read()

            if not ok:
                continue

            # Drop the oldest frame if the consumer is slow
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)
