"""Milestone 3: Scale-invariant gesture math, EMA smoothing, and state machines.

Key computer-vision & signal-processing ideas implemented here:
1. Aspect-corrected, palm-normalized distances (scale invariance across depth).
2. Hysteresis (Schmitt trigger) on pinch detection to eliminate threshold flicker.
3. Rotation-tolerant finger extension (wrist-to-tip vs. wrist-to-PIP distance).
4. Relative drag + Exponential Moving Average (EMA) smoothing for analog sliders.
5. Velocity rolling window + cooldowns + edge-triggered hold timers to stop misfires.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from hyprwave.tracker import Point3D, TrackedHand


class GestureType(Enum):
    NONE = auto()
    VOLUME_SLIDER = auto()      # Right hand pinch + vertical drag
    BRIGHTNESS_SLIDER = auto()  # Left hand pinch + vertical drag
    SWIPE_LEFT = auto()         # Fast right-to-left swipe -> next workspace
    SWIPE_RIGHT = auto()        # Fast left-to-right swipe -> previous workspace
    PALM_HOLD = auto()          # Open palm held 0.6s -> play/pause
    PEACE_HOLD = auto()         # Peace sign held 0.6s -> toggle mute


@dataclass(slots=True)
class PinchVisual:
    """HUD metadata for an active pinch-and-drag analog control."""

    kind: GestureType
    handedness: str
    value: float                  # 0.0 .. 1.0
    thumb_xy: tuple[float, float]
    index_xy: tuple[float, float]


@dataclass(slots=True)
class HoldVisual:
    """HUD metadata for a hold-to-trigger pose progress ring."""

    kind: GestureType
    label: str
    center_xy: tuple[float, float]
    progress: float               # 0.0 .. 1.0


@dataclass(slots=True)
class GestureDebug:
    """Live math telemetry displayed when the user presses 'd'."""

    handedness: str = "-"
    palm_size: float = 0.0
    pinch_ratio: float = 0.0
    fingers: str = "----"
    vx: float = 0.0
    cooldown_left: float = 0.0


@dataclass(slots=True)
class GestureEvent:
    """Per-frame output consumed by both the HUD renderer and ActionDispatcher."""

    kind: GestureType = GestureType.NONE
    triggered: bool = False       # True only on the single frame a discrete gesture fires
    value: float | None = None    # Continuous 0.0..1.0 value for analog sliders
    label: str = "SHOW A HAND TO START TRACKING"

    # Rich HUD overlay visuals
    pinch: PinchVisual | None = None
    hold: HoldVisual | None = None
    swipe_trail: list[tuple[float, float]] = field(default_factory=list)
    swipe_streak: list[tuple[float, float]] = field(default_factory=list)
    swipe_streak_alpha: float = 0.0
    swipe_direction: int = 0      # -1 = left, +1 = right, 0 = none
    flash_text: str | None = None
    flash_alpha: float = 0.0      # 0.0 .. 1.0 fade-out toast
    debug: GestureDebug = field(default_factory=GestureDebug)


def dist2d(a: Point3D, b: Point3D, aspect: float) -> float:
    """Aspect-corrected Euclidean distance in normalized frame coordinates.

    Because 640x480 has a 4:3 aspect ratio, multiplying dx by `aspect` ensures
    horizontal and vertical distances share the same physical scale.
    """
    dx = (a.x - b.x) * aspect
    dy = a.y - b.y
    return math.hypot(dx, dy)


def finger_states(hand: TrackedHand, aspect: float) -> tuple[bool, bool, bool, bool]:
    """Return `(index, middle, ring, pinky)` extended booleans.

    Comparing `dist(wrist, tip)` vs. `dist(wrist, PIP)` works regardless of how
    the hand is rotated in the image plane (unlike naive `tip.y < pip.y` checks).
    """
    lm = hand.landmarks
    wrist = lm[0]
    pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended: list[bool] = []
    for tip_idx, pip_idx in pairs:
        tip_d = dist2d(wrist, lm[tip_idx], aspect)
        pip_d = dist2d(wrist, lm[pip_idx], aspect)
        extended.append(tip_d > pip_d * 1.15)
    return (extended[0], extended[1], extended[2], extended[3])


def _is_curled(hand: TrackedHand, tip_idx: int, pip_idx: int, aspect: float) -> bool:
    lm = hand.landmarks
    wrist = lm[0]
    return dist2d(wrist, lm[tip_idx], aspect) < dist2d(wrist, lm[pip_idx], aspect) * 1.06


@dataclass
class _PinchState:
    active: bool = False
    handedness: str = ""
    anchor_y: float = 0.5
    anchor_value: float = 0.5
    smooth_y: float = 0.5


class GestureRecognizer:
    """Stateful gesture recognizer turning 21-point landmarks into clean events."""

    # Pinch hysteresis thresholds (normalized by palm length)
    PINCH_ENTER_RATIO = 0.36
    PINCH_EXIT_RATIO = 0.52
    # Vertical drag range (fraction of frame height) covering 0% -> 100%
    DRAG_SPAN_Y = 0.38
    EMA_ALPHA = 0.35

    # Hold-to-trigger timing
    HOLD_DURATION_S = 0.60
    HOLD_COOLDOWN_S = 0.90

    # Swipe detection tuning (calibrated for ~15-20 FPS webcam tracking)
    SWIPE_WINDOW_S = 0.45
    SWIPE_MIN_DT = 0.04
    SWIPE_MIN_DX = 0.14           # At least 14% of frame width
    SWIPE_MIN_SPEED = 0.45        # Frame-widths per second
    SWIPE_HORIZ_RATIO = 1.25      # Allows natural forearm arc while rejecting vertical moves
    SWIPE_COOLDOWN_S = 0.70
    HAND_LOST_GRACE_S = 0.30      # Survive several motion-blurred frames during fast swipes

    FLASH_DURATION_S = 1.15
    STREAK_DURATION_S = 0.70

    def __init__(
        self,
        aspect: float = 640.0 / 480.0,
        initial_volume: float = 0.50,
        initial_brightness: float = 0.70,
    ) -> None:
        self.aspect = aspect
        self.volume = max(0.0, min(1.0, initial_volume))
        self.brightness = max(0.0, min(1.0, initial_brightness))

        self._pinch = _PinchState()

        # Hold state (edge-triggered: must release pose before triggering again)
        self._hold_kind: GestureType = GestureType.NONE
        self._hold_start: float = 0.0
        self._hold_locked: bool = False
        self._hold_cooldown_until: float = 0.0

        # Rolling palm-center history for swipe velocity: (timestamp, x, y)
        self._wrist_history: deque[tuple[float, float, float]] = deque()
        self._swipe_cooldown_until: float = 0.0
        self._last_seen_time: float = 0.0

        # Persisted glowing comet streak after a swipe fires
        self._swipe_streak: list[tuple[float, float]] = []
        self._swipe_streak_until: float = 0.0
        self._swipe_direction: int = 0

        # Short-lived HUD confirmation flash
        self._flash_text: str | None = None
        self._flash_until: float = 0.0

    def _set_flash(self, text: str, now: float) -> None:
        self._flash_text = text
        self._flash_until = now + self.FLASH_DURATION_S

    def _current_flash(self, now: float) -> tuple[str | None, float]:
        if self._flash_text is None or now >= self._flash_until:
            self._flash_text = None
            return None, 0.0
        remaining = self._flash_until - now
        alpha = max(0.0, min(1.0, remaining / self.FLASH_DURATION_S))
        return self._flash_text, alpha

    def _current_streak(self, now: float) -> tuple[list[tuple[float, float]], float, int]:
        if not self._swipe_streak or now >= self._swipe_streak_until:
            self._swipe_streak = []
            self._swipe_direction = 0
            return [], 0.0, 0
        alpha = max(0.0, min(1.0, (self._swipe_streak_until - now) / self.STREAK_DURATION_S))
        return self._swipe_streak, alpha, self._swipe_direction

    def _reset_transient(self) -> None:
        self._pinch.active = False
        self._hold_kind = GestureType.NONE
        self._hold_locked = False
        self._wrist_history.clear()

    def update(self, hands: list[TrackedHand], now: float) -> GestureEvent:
        flash_text, flash_alpha = self._current_flash(now)
        streak_pts, streak_alpha, streak_dir = self._current_streak(now)

        if not hands:
            # Fast swipes can cause 1-2 frames of motion blur; wait briefly
            # before clearing the trail so a single missed frame doesn't abort
            # an in-progress swipe.
            if (now - self._last_seen_time) > self.HAND_LOST_GRACE_S:
                self._reset_transient()
            return GestureEvent(
                label="SHOW A HAND TO START TRACKING",
                swipe_trail=[(x, y) for _, x, y in self._wrist_history],
                swipe_streak=streak_pts,
                swipe_streak_alpha=streak_alpha,
                swipe_direction=streak_dir,
                flash_text=flash_text,
                flash_alpha=flash_alpha,
            )

        self._last_seen_time = now

        # Pick the primary hand: if currently pinching, stick with that hand so
        # a second hand entering frame doesn't hijack the active slider.
        hand = self._select_primary_hand(hands)
        lm = hand.landmarks
        palm_size = max(dist2d(lm[0], lm[9], self.aspect), 1e-4)
        pinch_ratio = dist2d(lm[4], lm[8], self.aspect) / palm_size
        idx_up, mid_up, ring_up, pinky_up = finger_states(hand, self.aspect)

        fingers_str = (
            ("I" if idx_up else "i")
            + ("M" if mid_up else "m")
            + ("R" if ring_up else "r")
            + ("P" if pinky_up else "p")
        )
        debug = GestureDebug(
            handedness=hand.handedness,
            palm_size=palm_size,
            pinch_ratio=pinch_ratio,
            fingers=fingers_str,
            cooldown_left=max(0.0, self._swipe_cooldown_until - now),
        )

        # ------------------------------------------------------------------
        # 1. Analog Pinch-and-Drag (Right = Volume, Left = Brightness)
        # ------------------------------------------------------------------
        mid_x = 0.5 * (lm[4].x + lm[8].x)
        mid_y = 0.5 * (lm[4].y + lm[8].y)

        if self._pinch.active and self._pinch.handedness == hand.handedness:
            if pinch_ratio > self.PINCH_EXIT_RATIO:
                self._pinch.active = False
                # Brief cooldown after releasing a slider so moving your hand
                # away doesn't accidentally fire a swipe.
                self._swipe_cooldown_until = max(self._swipe_cooldown_until, now + 0.35)
        elif pinch_ratio < self.PINCH_ENTER_RATIO:
            # Enter pinch state and anchor relative drag to current Y + value
            self._pinch.active = True
            self._pinch.handedness = hand.handedness
            self._pinch.anchor_y = mid_y
            self._pinch.smooth_y = mid_y
            self._pinch.anchor_value = self.volume if hand.handedness == "Right" else self.brightness
            # Cancel any in-progress hold or swipe trail while dragging
            self._hold_kind = GestureType.NONE
            self._wrist_history.clear()

        if self._pinch.active and self._pinch.handedness == hand.handedness:
            self._pinch.smooth_y = (
                (1.0 - self.EMA_ALPHA) * self._pinch.smooth_y + self.EMA_ALPHA * mid_y
            )
            delta_val = (self._pinch.anchor_y - self._pinch.smooth_y) / self.DRAG_SPAN_Y
            new_val = max(0.0, min(1.0, self._pinch.anchor_value + delta_val))

            if hand.handedness == "Right":
                self.volume = new_val
                kind = GestureType.VOLUME_SLIDER
                label = f"VOLUME: {int(round(new_val * 100))}%"
            else:
                self.brightness = new_val
                kind = GestureType.BRIGHTNESS_SLIDER
                label = f"BRIGHTNESS: {int(round(new_val * 100))}%"

            return GestureEvent(
                kind=kind,
                triggered=True,
                value=new_val,
                label=label,
                pinch=PinchVisual(
                    kind=kind,
                    handedness=hand.handedness,
                    value=new_val,
                    thumb_xy=(lm[4].x, lm[4].y),
                    index_xy=(lm[8].x, lm[8].y),
                ),
                flash_text=flash_text,
                flash_alpha=flash_alpha,
                debug=debug,
            )

        # ------------------------------------------------------------------
        # 2. Swipe Velocity Detection (rolling knuckle-centroid history + cooldown)
        # ------------------------------------------------------------------
        # Average the 4 knuckle bases (#5, #9, #13, #17): rock-steady even when
        # the wrist/forearm dips near the bottom edge of the camera frame.
        anchor_x = 0.25 * (lm[5].x + lm[9].x + lm[13].x + lm[17].x)
        anchor_y = 0.25 * (lm[5].y + lm[9].y + lm[13].y + lm[17].y)
        self._wrist_history.append((now, anchor_x, anchor_y))
        while self._wrist_history and (now - self._wrist_history[0][0]) > self.SWIPE_WINDOW_S:
            self._wrist_history.popleft()

        trail = [(x, y) for _, x, y in self._wrist_history]
        vx = 0.0
        best_swipe_dx = 0.0

        # Scan all prior points in the rolling window so starting from rest
        # (or surviving a 1-2 frame blur gap) still captures peak swipe speed.
        if len(self._wrist_history) >= 2:
            best_speed = 0.0
            for t0, x0, y0 in list(self._wrist_history)[:-1]:
                dt = now - t0
                if dt < self.SWIPE_MIN_DT:
                    continue
                dx = anchor_x - x0
                dy = anchor_y - y0
                speed = abs(dx) / dt
                if speed > best_speed:
                    best_speed = speed
                    vx = dx / dt
                if (
                    now >= self._swipe_cooldown_until
                    and abs(dx) >= self.SWIPE_MIN_DX
                    and speed >= self.SWIPE_MIN_SPEED
                    and abs(dx) >= self.SWIPE_HORIZ_RATIO * max(abs(dy), 1e-4)
                ):
                    if abs(dx) > abs(best_swipe_dx):
                        best_swipe_dx = dx

            debug.vx = vx

            if best_swipe_dx != 0.0:
                self._swipe_cooldown_until = now + self.SWIPE_COOLDOWN_S
                self._wrist_history.clear()
                self._hold_kind = GestureType.NONE

                if best_swipe_dx < 0:
                    kind = GestureType.SWIPE_LEFT
                    direction = -1
                    msg = "<<<  SWIPE LEFT  (NEXT WS)"
                else:
                    kind = GestureType.SWIPE_RIGHT
                    direction = 1
                    msg = "SWIPE RIGHT  (PREV WS)  >>>"

                self._swipe_streak = list(trail)
                self._swipe_streak_until = now + self.STREAK_DURATION_S
                self._swipe_direction = direction
                self._set_flash(msg, now)
                return GestureEvent(
                    kind=kind,
                    triggered=True,
                    label=msg,
                    swipe_trail=trail,
                    swipe_streak=self._swipe_streak,
                    swipe_streak_alpha=1.0,
                    swipe_direction=direction,
                    flash_text=msg,
                    flash_alpha=1.0,
                    debug=debug,
                )

        # ------------------------------------------------------------------
        # 3. Held Static Poses (Open Palm = Play/Pause, Peace = Toggle Mute)
        # ------------------------------------------------------------------
        # Suppress hold detection while the hand is moving rapidly horizontally.
        hand_is_steady = abs(vx) < 0.45
        detected_pose = GestureType.NONE
        pose_label = ""

        if hand_is_steady and pinch_ratio > self.PINCH_EXIT_RATIO:
            if idx_up and mid_up and ring_up and pinky_up:
                detected_pose = GestureType.PALM_HOLD
                pose_label = "PLAY / PAUSE"
            elif (
                idx_up
                and mid_up
                and _is_curled(hand, 16, 14, self.aspect)
                and _is_curled(hand, 20, 18, self.aspect)
                and (dist2d(lm[8], lm[12], self.aspect) / palm_size) > 0.22
            ):
                detected_pose = GestureType.PEACE_HOLD
                pose_label = "TOGGLE MUTE"

        if detected_pose == GestureType.NONE:
            self._hold_kind = GestureType.NONE
            self._hold_locked = False
        elif detected_pose != self._hold_kind:
            self._hold_kind = detected_pose
            self._hold_start = now
            self._hold_locked = False
        elif not self._hold_locked and now >= self._hold_cooldown_until:
            elapsed = now - self._hold_start
            progress = max(0.0, min(1.0, elapsed / self.HOLD_DURATION_S))
            palm_center = (lm[9].x, lm[9].y)

            if progress >= 1.0:
                self._hold_locked = True
                self._hold_cooldown_until = now + self.HOLD_COOLDOWN_S
                msg = f"{pose_label} TRIGGERED"
                self._set_flash(msg, now)
                return GestureEvent(
                    kind=detected_pose,
                    triggered=True,
                    label=msg,
                    hold=HoldVisual(detected_pose, pose_label, palm_center, 1.0),
                    swipe_trail=trail,
                    flash_text=msg,
                    flash_alpha=1.0,
                    debug=debug,
                )

            return GestureEvent(
                kind=detected_pose,
                triggered=False,
                label=f"HOLDING {pose_label}... {int(progress * 100)}%",
                hold=HoldVisual(detected_pose, pose_label, palm_center, progress),
                swipe_trail=trail,
                flash_text=flash_text,
                flash_alpha=flash_alpha,
                debug=debug,
            )

        # Default idle tracking state
        status = f"TRACKING {hand.handedness.upper()}  |  PINCH / SWIPE / PALM / PEACE"
        if now < self._swipe_cooldown_until:
            status = f"COOLDOWN ({self._swipe_cooldown_until - now:.1f}s)"

        return GestureEvent(
            kind=GestureType.NONE,
            label=status,
            swipe_trail=trail,
            swipe_streak=streak_pts,
            swipe_streak_alpha=streak_alpha,
            swipe_direction=streak_dir,
            flash_text=flash_text,
            flash_alpha=flash_alpha,
            debug=debug,
        )

    def _select_primary_hand(self, hands: list[TrackedHand]) -> TrackedHand:
        if self._pinch.active:
            for h in hands:
                if h.handedness == self._pinch.handedness:
                    return h
        return max(hands, key=lambda h: h.score)
