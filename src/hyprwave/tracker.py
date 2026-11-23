"""Milestone 2: MediaPipe Hand Landmarker wrapper (Tasks API, VIDEO mode)."""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


# Desktop display/session vars that cause MediaPipe's LiteRT runtime to probe
# Wayland/X11 EGL surfaces during model init (which can get SIGKILL'd on Intel
# Kaby Lake + Hyprland even when Delegate.CPU is selected). Hiding them solely
# during `create_from_options()` keeps LiteRT cleanly on XNNPACK CPU while
# leaving the environment intact for our native Wayland PyQt6 HUD window.
_HIDDEN_DISPLAY_VARS: tuple[str, ...] = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
)


@contextlib.contextmanager
def _headless_env() -> Iterator[None]:
    saved = {k: os.environ.pop(k) for k in _HIDDEN_DISPLAY_VARS if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


# Standard 21-landmark MediaPipe hand topology: pairs of landmark indices
# forming the palm and five fingers.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)


@dataclass(slots=True)
class Point3D:
    """Normalized landmark coordinates: x, y in [0, 1], z relative to wrist."""

    x: float
    y: float
    z: float


@dataclass(slots=True)
class TrackedHand:
    """A single detected hand with its 21 landmarks and handedness label."""

    landmarks: list[Point3D]
    handedness: str  # "Left" or "Right"
    score: float

    def pixel_point(self, index: int, width: int, height: int) -> tuple[int, int]:
        """Convert normalized landmark `index` into `(px, py)` pixel coords."""
        lm = self.landmarks[index]
        return (int(lm.x * width), int(lm.y * height))


class HandTracker:
    """Wraps MediaPipe's modern `HandLandmarker` in `RunningMode.VIDEO`.

    Why VIDEO mode instead of IMAGE mode?
    In IMAGE mode, MediaPipe runs the palm detector on every single frame.
    In VIDEO mode, once a hand is found, MediaPipe tracks it frame-to-frame
    using the previous bounding box—significantly faster and cooler on CPU.
    """

    def __init__(
        self,
        model_path: Path,
        num_hands: int = 2,
        min_detection_confidence: float = 0.55,
        min_presence_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
        mirrored: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mirrored = mirrored

        self._landmarker = None
        self._last_timestamp_ms: int = 0

    def open(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found at: {self.model_path}\n"
                "Run `./scripts/fetch_model.sh` to download hand_landmarker.task."
            )

        with _headless_env():
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._mp = mp

            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.model_path),
                    delegate=mp_python.BaseOptions.Delegate.CPU,
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def process(self, bgr_frame: np.ndarray) -> tuple[list[TrackedHand], float]:
        """Run inference on a BGR frame; returns `(hands, inference_ms)`."""
        if self._landmarker is None:
            raise RuntimeError("HandTracker is not open. Call open() or use a 'with' block.")

        # MediaPipe requires RGB; OpenCV captures BGR.
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode requires strictly increasing integer millisecond timestamps.
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms

        t0 = time.perf_counter()
        result = self._landmarker.detect_for_video(mp_image, now_ms)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        hands: list[TrackedHand] = []
        if not result.hand_landmarks:
            return hands, infer_ms

        for idx, raw_landmarks in enumerate(result.hand_landmarks):
            points = [Point3D(lm.x, lm.y, lm.z) for lm in raw_landmarks]
            label = "Hand"
            score = 0.0
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                cat = result.handedness[idx][0]
                label = cat.category_name
                # MediaPipe classifies handedness assuming a non-mirrored view
                # (like someone facing you: their left hand is on the right side
                # with the thumb inward). Since we mirror the webcam frame for
                # natural HUD feedback, swap Left <-> Right to match the user's
                # actual hands.
                if self.mirrored:
                    if label == "Left":
                        label = "Right"
                    elif label == "Right":
                        label = "Left"
                score = float(cat.score)
            hands.append(TrackedHand(landmarks=points, handedness=label, score=score))

        return hands, infer_ms

    def __enter__(self) -> HandTracker:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
