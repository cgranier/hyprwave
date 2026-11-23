"""Native Wayland HUD window built with PyQt6.

Why PyQt6 instead of `cv2.imshow()`?
PyPI's OpenCV wheels bundle Qt5 with *only* the X11 (`xcb`) platform plugin.
On a pure-Wayland Hyprland desktop (with no X11 `DISPLAY`), `cv2.namedWindow()`
aborts immediately. PyQt6 ships native Qt6 Wayland support (`libqwayland-*`),
scales cleanly on the Pixelbook's 2400x1600 HiDPI display, and gives us a real
event loop for smooth resizing and future overlay/tray features.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
class ViewFlags:
    """Interactive HUD toggles controlled via keyboard shortcuts."""

    show_indices: bool = False
    show_debug: bool = False
    dry_run: bool = True


def _sanitize_qt_env() -> None:
    """Undo OpenCV's Qt5 plugin env overrides so PyQt6 uses its own Qt6 plugins.

    When `cv2` is imported (pulled in by MediaPipe), its `config-3.py` sets
    `QT_QPA_PLATFORM_PLUGIN_PATH` and `QT_QPA_FONTDIR` to OpenCV's bundled Qt5
    directories. Leaving those set causes PyQt6 to fail to load Wayland.
    """
    for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        val = os.environ.get(var, "")
        if "cv2" in val:
            os.environ.pop(var, None)


def run_hud_window(
    title: str,
    width: int,
    height: int,
    target_fps: float,
    next_frame: Callable[[ViewFlags], tuple[np.ndarray, float]],
    initial_flags: ViewFlags | None = None,
) -> int:
    """Launch the native Wayland PyQt6 HUD window and run its adaptive frame loop."""
    _sanitize_qt_env()

    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
    from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

    flags = initial_flags or ViewFlags()

    def _fps_to_ms(fps: float) -> int:
        return max(1, int(round(1000.0 / fps))) if fps > 0 else 16

    class HudWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(title)
            self.setStyleSheet("background-color: #14161a;")
            self.resize(width, height)

            self.flags = flags
            self._last_pixmap: QPixmap | None = None
            self._current_interval_ms = _fps_to_ms(target_fps)

            self._label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
            self._label.setMinimumSize(320, 240)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._label)

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._on_tick)
            self._timer.start(self._current_interval_ms)

        def _on_tick(self) -> None:
            try:
                bgr_frame, desired_fps = next_frame(self.flags)
            except Exception as exc:  # pragma: no cover - runtime camera error
                print(f"[hyprwave] Frame error: {exc}")
                self.close()
                return

            # Adaptive FPS: throttle down when idle, snap back immediately on hand entry.
            next_interval = _fps_to_ms(desired_fps)
            if next_interval != self._current_interval_ms:
                self._current_interval_ms = next_interval
                self._timer.setInterval(next_interval)

            h, w, ch = bgr_frame.shape
            bytes_per_line = ch * w
            # Ensure contiguous memory before wrapping with QImage.
            contiguous = np.ascontiguousarray(bgr_frame)
            qimg = QImage(contiguous.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).copy()
            self._last_pixmap = QPixmap.fromImage(qimg)
            self._render_pixmap()

        def _render_pixmap(self) -> None:
            if self._last_pixmap is None:
                return
            scaled = self._last_pixmap.scaled(
                self._label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self._label.setPixmap(scaled)

        def resizeEvent(self, event) -> None:  # type: ignore[override]
            super().resizeEvent(event)
            self._render_pixmap()

        def keyPressEvent(self, event: QKeyEvent | None) -> None:
            if event is None:
                return
            key = event.key()
            if key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
                self.close()
            elif key == Qt.Key.Key_I:
                self.flags.show_indices = not self.flags.show_indices
            elif key == Qt.Key.Key_D:
                self.flags.show_debug = not self.flags.show_debug
            elif key == Qt.Key.Key_L:
                self.flags.dry_run = not self.flags.dry_run

        def closeEvent(self, event) -> None:  # type: ignore[override]
            self._timer.stop()
            super().closeEvent(event)

    app = QApplication([])
    app.setApplicationName(title)
    app.setDesktopFileName("hyprwave")

    # Allow Ctrl+C or SIGTERM (from toggle_hyprwave.sh) to close cleanly and release the webcam.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    sig_timer = QTimer()
    sig_timer.start(200)
    sig_timer.timeout.connect(lambda: None)

    window = HudWindow()
    window.show()
    _ = time.monotonic()  # keeps import used if needed later
    return app.exec()
