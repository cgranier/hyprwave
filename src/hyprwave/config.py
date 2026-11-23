"""Configuration defaults tuned for a fanless Pixelbook (i7-7Y75)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


@dataclass(slots=True)
class AppConfig:
    """Runtime settings for capture, tracking, and HUD rendering."""

    # Camera settings:
    # The Pixelbook UVC webcam natively outputs 640x480 YUYV @ 30 FPS
    # with zero JPEG decoding overhead. We pace inference to 20 FPS to
    # keep the fanless dual-core CPU cool.
    device_index: int = 0
    width: int = 640
    height: int = 480
    target_fps: float = 20.0
    idle_fps: float = 4.0
    idle_after_s: float = 2.5
    mirror: bool = True

    # MediaPipe Hand Landmarker settings:
    # Lower tracking confidence (0.35) helps MediaPipe stay locked onto fast
    # swipes through webcam motion blur without dropping the hand.
    model_path: Path = DEFAULT_MODEL_PATH
    enable_tracking: bool = True
    num_hands: int = 2
    min_detection_confidence: float = 0.50
    min_presence_confidence: float = 0.45
    min_tracking_confidence: float = 0.35

    # HUD settings:
    window_title: str = "HyprWave"
    show_landmarks_index: bool = False
    show_debug_math: bool = False
    dry_run: bool = True
