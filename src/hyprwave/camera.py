"""Milestone 1: V4L2/OpenCV webcam capture with frame pacing."""

from __future__ import annotations

import time
from types import TracebackType

import cv2
import numpy as np


class Camera:
    """Thin wrapper around OpenCV VideoCapture with FPS pacing and mirroring.

    Why pace frames manually?
    On a fanless laptop (like the Pixelbook i7-7Y75), running a tight capture +
    inference loop at 30+ FPS burns unnecessary CPU cycles and heat. Sleeping
    the remainder of each frame budget keeps the HUD responsive at ~15–20 FPS
    while leaving plenty of thermal headroom.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: float = 20.0,
        mirror: bool = True,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.mirror = mirror

        self._cap: cv2.VideoCapture | None = None
        self._last_frame_time: float = 0.0
        self._fps_ema: float = 0.0

    def open(self) -> None:
        # CAP_V4L2 avoids backend probing delays on Linux.
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera /dev/video{self.device_index}. "
                "Check that no other app is using the webcam and your user is in the 'video' group."
            )

        # Request 640x480 uncompressed YUYV (native on Pixelbook Eve @ 30 FPS)
        # and keep the capture buffer shallow (1 frame) to minimize latency.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._last_frame_time = time.perf_counter()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self, pace: bool = True) -> tuple[np.ndarray, float]:
        """Read the next frame (BGR) and return `(frame, smoothed_fps)`."""
        if self._cap is None:
            raise RuntimeError("Camera is not open. Call open() or use a 'with' block.")

        # Frame pacing: sleep if we're ahead of our target frame interval
        # (disabled when an external event loop like Qt's QTimer paces frames).
        if pace and self.target_fps > 0:
            min_interval = 1.0 / self.target_fps
            elapsed = time.perf_counter() - self._last_frame_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        ok, frame = self._cap.read()
        now = time.perf_counter()
        dt = max(now - self._last_frame_time, 1e-6)
        self._last_frame_time = now

        instant_fps = 1.0 / dt
        self._fps_ema = instant_fps if self._fps_ema == 0.0 else (0.85 * self._fps_ema + 0.15 * instant_fps)

        if not ok or frame is None:
            raise RuntimeError("Failed to read a frame from the webcam.")

        # Mirror horizontally so moving your hand right moves it right on screen.
        # Doing this *before* MediaPipe inference also makes handedness labels
        # intuitive from the user's point of view.
        if self.mirror:
            frame = cv2.flip(frame, 1)

        return frame, self._fps_ema

    @property
    def actual_resolution(self) -> tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
