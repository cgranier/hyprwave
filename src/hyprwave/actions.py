"""Milestone 4: Non-blocking, rate-limited Hyprland, PipeWire & Omarchy IPC.

Design goals:
- Safe by default (`dry_run=True` -> DEMO mode until the user presses 'l' or passes `--live`).
- Never block the 17-20 FPS HUD render loop: commands spawn asynchronously and are
  rate-limited so continuous pinch sliders don't flood `wpctl` or `brightnessctl`.
- Uses Omarchy's native commands where available (`omarchy-shell -q media playPause`,
  `omarchy-audio-output-volume mute-toggle`) and clamps brightness to >= 5% so a
  pinch can never accidentally black out the screen.
"""

from __future__ import annotations

import shutil
import subprocess
import time

from hyprwave.gestures import GestureEvent, GestureType


def read_system_volume(default: float = 0.50) -> float:
    """Read current PipeWire default sink volume in [0.0, 1.0]."""
    if not shutil.which("wpctl"):
        return default
    try:
        res = subprocess.run(
            ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        # Format: "Volume: 0.65" or "Volume: 0.65 [MUTED]"
        for token in res.stdout.split():
            try:
                val = float(token)
                return max(0.0, min(1.0, val))
            except ValueError:
                continue
    except Exception:
        pass
    return default


def read_system_brightness(default: float = 0.70) -> float:
    """Read current internal panel brightness in [0.05, 1.0]."""
    if not shutil.which("brightnessctl"):
        return default
    try:
        res = subprocess.run(
            ["brightnessctl", "-d", "intel_backlight", "-m"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        # Format: "intel_backlight,backlight,22937,70%,32767"
        parts = res.stdout.strip().split(",")
        if len(parts) >= 4 and parts[3].endswith("%"):
            pct = float(parts[3][:-1]) / 100.0
            return max(0.05, min(1.0, pct))
    except Exception:
        pass
    return default


class ActionDispatcher:
    """Dispatches recognized gestures to Hyprland, PipeWire, and backlight controls."""

    SLIDER_MIN_INTERVAL_S = 0.08  # At most ~12 IPC updates/sec while pinching
    MIN_BRIGHTNESS_PCT = 5        # Never black out the screen completely

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self._last_vol_time: float = 0.0
        self._last_vol_pct: int = -1
        self._last_brt_time: float = 0.0
        self._last_brt_pct: int = -1
        self._pending: list[subprocess.Popen[bytes]] = []

    def _spawn(self, cmd: list[str]) -> None:
        """Launch a desktop IPC command without blocking the render loop."""
        # Reap finished child processes so we never accumulate zombies.
        self._pending = [p for p in self._pending if p.poll() is None]
        if not shutil.which(cmd[0]):
            return
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._pending.append(proc)
        except OSError:
            pass

    def dispatch(self, event: GestureEvent, now: float | None = None) -> None:
        if self.dry_run or not event.triggered:
            return

        ts = time.monotonic() if now is None else now
        kind = event.kind

        if kind == GestureType.SWIPE_LEFT:
            # Omarchy's Lua-based Hyprland takes a Lua dispatcher expression.
            # Use "r+1"/"r-1" to step through relative workspaces (works even if
            # only 1 workspace is open), or swap to "m+1"/"m-1" to cycle only
            # existing workspaces on the current monitor like Super+Tab.
            self._spawn(["hyprctl", "dispatch", 'hl.dsp.focus({ workspace = "r+1" })'])
        elif kind == GestureType.SWIPE_RIGHT:
            self._spawn(["hyprctl", "dispatch", 'hl.dsp.focus({ workspace = "r-1" })'])
        elif kind == GestureType.PALM_HOLD:
            self._spawn(["omarchy-shell", "-q", "media", "playPause"])
        elif kind == GestureType.PEACE_HOLD:
            if shutil.which("omarchy-audio-output-volume"):
                self._spawn(["omarchy-audio-output-volume", "mute-toggle"])
            else:
                self._spawn(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"])
        elif kind == GestureType.VOLUME_SLIDER and event.value is not None:
            pct = int(round(event.value * 100))
            if (ts - self._last_vol_time) >= self.SLIDER_MIN_INTERVAL_S and abs(pct - self._last_vol_pct) >= 2:
                self._last_vol_time = ts
                self._last_vol_pct = pct
                self._spawn(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{pct / 100.0:.2f}"])
        elif kind == GestureType.BRIGHTNESS_SLIDER and event.value is not None:
            pct = max(self.MIN_BRIGHTNESS_PCT, int(round(event.value * 100)))
            if (ts - self._last_brt_time) >= self.SLIDER_MIN_INTERVAL_S and abs(pct - self._last_brt_pct) >= 2:
                self._last_brt_time = ts
                self._last_brt_pct = pct
                self._spawn(["brightnessctl", "-d", "intel_backlight", "set", f"{pct}%"])
