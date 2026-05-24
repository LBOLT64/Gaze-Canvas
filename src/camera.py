"""
Gaze Canvas — threaded webcam capture with a bounded frame queue.
"""

import threading
import queue

import cv2
import numpy as np


class CameraCapture:
    """Continuously grabs webcam frames on a background daemon thread."""

    def __init__(self, device_id: int = 0) -> None:
        self._cap = cv2.VideoCapture(device_id)
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Begin the background capture thread."""
        self._stop_event.clear()
        self._thread.start()

    def get_frame(self) -> np.ndarray | None:
        """Return the latest frame without blocking, or *None*."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Signal the thread to stop and release the camera."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap.isOpened():
            self._cap.release()

    def is_open(self) -> bool:
        return self._cap.isOpened()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _reader(self) -> None:
        """Loop that fills the queue, dropping stale frames when full."""
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                continue
            if self._queue.full():
                try:
                    self._queue.get_nowait()          # discard oldest
                except queue.Empty:
                    pass
            self._queue.put(frame)
