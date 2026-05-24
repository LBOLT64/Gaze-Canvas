"""
Gaze Canvas — threaded webcam capture with a bounded frame queue.

Robustness guarantees
---------------------
* The reader thread is a daemon so it dies if the main process exits.
* ``stop()`` is safe to call multiple times.
* If the webcam disconnects mid-run, ``get_frame()`` simply returns None
  and ``is_open()`` becomes False — no crash.
"""

from __future__ import annotations

import logging
import queue
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    """Continuously grabs webcam frames on a background daemon thread."""

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id
        self._cap: cv2.VideoCapture | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()          # guards _cap access

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> bool:
        """Open the camera and start the reader thread.

        Returns True if the webcam opened successfully.
        """
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            logger.error("Could not open webcam (device %s)", self._device_id)
            return False

        logger.info(
            "Webcam opened — %.0fx%.0f @ %.0f fps",
            self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            self._cap.get(cv2.CAP_PROP_FPS),
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
        """Signal the thread to stop and release the camera.  Safe to call
        repeatedly or from any thread."""
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
