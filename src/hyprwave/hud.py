"""Milestones 1–3: Glowing neon hand skeleton, sliders, hold rings, and gesture HUD."""

from __future__ import annotations

import cv2
import numpy as np

from hyprwave.gestures import GestureEvent, GestureType, HoldVisual, PinchVisual
from hyprwave.tracker import FINGERTIP_INDICES, HAND_CONNECTIONS, TrackedHand

# Tokyo Night / Omarchy-inspired BGR color palette
BG_PANEL = (26, 22, 20)
ACCENT_CYAN = (247, 207, 125)     # Cyan-blue glow (Left hand / Brightness)
ACCENT_GREEN = (106, 206, 158)    # Omarchy green (#9ece6a in BGR, Right hand / Volume)
ACCENT_MAGENTA = (247, 154, 187)  # Soft violet/magenta (Fingertips / Pinch)
ACCENT_AMBER = (104, 175, 224)    # Warm amber (Hold ring / Flash)
TEXT_PRIMARY = (245, 240, 235)
TEXT_MUTED = (175, 165, 155)


def draw_hud(
    frame: np.ndarray,
    hands: list[TrackedHand],
    *,
    fps: float,
    infer_ms: float | None = None,
    event: GestureEvent | None = None,
    status_text: str | None = None,
    show_indices: bool = False,
    show_debug: bool = False,
    dry_run: bool = True,
    is_idle: bool = False,
) -> np.ndarray:
    """Render the glowing hand skeleton and Milestone 3 gesture HUD onto `frame`."""
    out = frame.copy()
    h, w = out.shape[:2]

    glow_layer = np.zeros_like(out)
    has_glow = False

    # 1. Live wrist/palm motion trail + persisted bright swipe comet streak
    if event and len(event.swipe_trail) >= 2:
        pts_trail = [(int(x * w), int(y * h)) for x, y in event.swipe_trail]
        n = len(pts_trail)
        for i in range(1, n):
            frac = i / n
            thickness = max(2, int(frac * 6))
            color = tuple(int(c * (0.35 + 0.65 * frac)) for c in ACCENT_CYAN)
            cv2.line(glow_layer, pts_trail[i - 1], pts_trail[i], color, thickness + 4, cv2.LINE_AA)
            cv2.line(out, pts_trail[i - 1], pts_trail[i], color, thickness, cv2.LINE_AA)
        has_glow = True

    if event and len(event.swipe_streak) >= 2 and event.swipe_streak_alpha > 0.05:
        alpha = event.swipe_streak_alpha
        pts_streak = [(int(x * w), int(y * h)) for x, y in event.swipe_streak]
        n = len(pts_streak)
        streak_color = tuple(int(c * alpha) for c in ACCENT_GREEN)
        core_color = tuple(int(c * alpha) for c in TEXT_PRIMARY)
        for i in range(1, n):
            frac = i / n
            cv2.line(glow_layer, pts_streak[i - 1], pts_streak[i], streak_color, int(6 + 12 * frac), cv2.LINE_AA)
            cv2.line(out, pts_streak[i - 1], pts_streak[i], core_color, max(2, int(5 * frac)), cv2.LINE_AA)
        # Bold directional arrowhead at the tip of the swipe
        cv2.arrowedLine(
            out,
            pts_streak[-2],
            pts_streak[-1],
            ACCENT_GREEN,
            4,
            cv2.LINE_AA,
            tipLength=0.45,
        )
        # Subtle glowing edge pulse on the swiped side of the HUD
        edge_x = 8 if event.swipe_direction < 0 else (w - 8)
        cv2.line(glow_layer, (edge_x, 48), (edge_x, h - 40), streak_color, 18, cv2.LINE_AA)
        cv2.line(out, (edge_x, 48), (edge_x, h - 40), streak_color, 4, cv2.LINE_AA)
        has_glow = True

    # 2. Hand skeletons with layered neon glow
    if hands:
        has_glow = True
        for hand in hands:
            color = ACCENT_GREEN if hand.handedness == "Right" else ACCENT_CYAN
            pts = [hand.pixel_point(i, w, h) for i in range(len(hand.landmarks))]

            for a, b in HAND_CONNECTIONS:
                cv2.line(glow_layer, pts[a], pts[b], color, thickness=6, lineType=cv2.LINE_AA)
                cv2.line(out, pts[a], pts[b], color, thickness=2, lineType=cv2.LINE_AA)

            for idx, pt in enumerate(pts):
                if idx in FINGERTIP_INDICES:
                    cv2.circle(glow_layer, pt, 10, ACCENT_MAGENTA, -1, lineType=cv2.LINE_AA)
                    cv2.circle(out, pt, 5, TEXT_PRIMARY, -1, lineType=cv2.LINE_AA)
                    cv2.circle(out, pt, 6, ACCENT_MAGENTA, 2, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(out, pt, 3, TEXT_PRIMARY, -1, lineType=cv2.LINE_AA)

                if show_indices:
                    cv2.putText(
                        out,
                        str(idx),
                        (pt[0] + 6, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        TEXT_PRIMARY,
                        1,
                        cv2.LINE_AA,
                    )

            wrist_x, wrist_y = pts[0]
            badge = f"{hand.handedness.upper()} {int(hand.score * 100)}%"
            _draw_pill(out, badge, (max(12, wrist_x - 36), min(h - 42, wrist_y + 26)), color)

    # 3. Active pinch connector + vertical analog slider bar
    if event and event.pinch is not None:
        has_glow = True
        _draw_pinch_overlay(out, glow_layer, event.pinch, w, h)

    # 4. Hold-to-trigger circular progress arc around palm
    if event and event.hold is not None:
        has_glow = True
        _draw_hold_ring(out, glow_layer, event.hold, w, h)

    # Blend blurred glow layer additively once for all glowing elements
    if has_glow:
        glow_blur = cv2.GaussianBlur(glow_layer, (15, 15), 0)
        cv2.addWeighted(out, 1.0, glow_blur, 0.45, 0, dst=out)

    # 5. Trigger flash toast banner
    if event and event.flash_text and event.flash_alpha > 0.05:
        _draw_flash_banner(out, event.flash_text, event.flash_alpha, w, h)

    # 6. Optional live gesture-math debug card (toggle with 'd')
    if show_debug and event is not None:
        _draw_debug_card(out, event, h)

    # 7. Top translucent telemetry bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, dst=out)

    mode_tag = "DEMO" if dry_run else "LIVE"
    mode_color = ACCENT_CYAN if dry_run else ACCENT_GREEN
    cv2.putText(out, "HYPRWAVE", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, ACCENT_GREEN, 2, cv2.LINE_AA)
    _draw_pill(out, mode_tag, (130, 26), mode_color)
    if is_idle:
        _draw_pill(out, "IDLE", (190, 26), ACCENT_AMBER)

    telemetry = f"{fps:4.1f} FPS"
    if infer_ms is not None:
        telemetry += f"  |  {infer_ms:4.1f} ms"
    telemetry += f"  |  HANDS: {len(hands)}"
    cv2.putText(out, telemetry, (w - 255, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_PRIMARY, 1, cv2.LINE_AA)

    # 8. Bottom status bar with hotkeys
    overlay_bot = out.copy()
    cv2.rectangle(overlay_bot, (0, h - 32), (w, h), BG_PANEL, -1)
    cv2.addWeighted(overlay_bot, 0.72, out, 0.28, 0, dst=out)

    footer = status_text if status_text is not None else (event.label if event else "TRACKING READY")
    cv2.putText(out, footer, (14, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_PRIMARY, 1, cv2.LINE_AA)
    cv2.putText(
        out,
        "[L] live  [I] idx  [D] math  [Q] quit",
        (w - 285, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    return out


def _draw_pinch_overlay(
    out: np.ndarray,
    glow_layer: np.ndarray,
    pinch: PinchVisual,
    w: int,
    h: int,
) -> None:
    """Draw pinch anchor spark and a sleek docked vertical slider bar."""
    accent = ACCENT_GREEN if pinch.kind == GestureType.VOLUME_SLIDER else ACCENT_CYAN
    title = "VOL" if pinch.kind == GestureType.VOLUME_SLIDER else "BRT"

    p_thumb = (int(pinch.thumb_xy[0] * w), int(pinch.thumb_xy[1] * h))
    p_index = (int(pinch.index_xy[0] * w), int(pinch.index_xy[1] * h))
    mid = ((p_thumb[0] + p_index[0]) // 2, (p_thumb[1] + p_index[1]) // 2)

    cv2.line(glow_layer, p_thumb, p_index, ACCENT_MAGENTA, 6, cv2.LINE_AA)
    cv2.line(out, p_thumb, p_index, TEXT_PRIMARY, 2, cv2.LINE_AA)
    cv2.circle(glow_layer, mid, 14, accent, -1, cv2.LINE_AA)
    cv2.circle(out, mid, 6, accent, -1, cv2.LINE_AA)

    # Dock volume slider on right edge, brightness slider on left edge
    bar_w = 24
    bar_h = 200
    bar_x = (w - 48) if pinch.kind == GestureType.VOLUME_SLIDER else 24
    bar_y = (h - bar_h) // 2

    # Translucent track background
    panel = out.copy()
    cv2.rectangle(panel, (bar_x - 10, bar_y - 30), (bar_x + bar_w + 10, bar_y + bar_h + 30), BG_PANEL, -1)
    cv2.addWeighted(panel, 0.75, out, 0.25, 0, dst=out)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), TEXT_MUTED, 1, cv2.LINE_AA)

    # Filled level from bottom up
    fill_h = int(round(bar_h * pinch.value))
    fill_top = bar_y + bar_h - fill_h
    if fill_h > 0:
        cv2.rectangle(glow_layer, (bar_x + 2, fill_top), (bar_x + bar_w - 2, bar_y + bar_h - 1), accent, -1)
        cv2.rectangle(out, (bar_x + 2, fill_top), (bar_x + bar_w - 2, bar_y + bar_h - 1), accent, -1)

    # Horizontal handle marker
    cv2.line(out, (bar_x - 4, fill_top), (bar_x + bar_w + 4, fill_top), TEXT_PRIMARY, 2, cv2.LINE_AA)

    pct_text = f"{int(round(pinch.value * 100))}%"
    cv2.putText(out, title, (bar_x - 2, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, accent, 2, cv2.LINE_AA)
    cv2.putText(out, pct_text, (bar_x - 6, bar_y + bar_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_PRIMARY, 1, cv2.LINE_AA)


def _draw_hold_ring(
    out: np.ndarray,
    glow_layer: np.ndarray,
    hold: HoldVisual,
    w: int,
    h: int,
) -> None:
    """Draw a circular progress arc around the palm while holding a pose."""
    cx = int(hold.center_xy[0] * w)
    cy = int(hold.center_xy[1] * h)
    radius = 44
    sweep_deg = int(round(360 * hold.progress))
    color = ACCENT_AMBER if hold.progress < 1.0 else ACCENT_GREEN

    # Subtle background ring
    cv2.circle(out, (cx, cy), radius, TEXT_MUTED, 1, cv2.LINE_AA)
    # Glowing progress arc starting at 12 o'clock (-90 deg)
    cv2.ellipse(glow_layer, (cx, cy), (radius, radius), -90, 0, sweep_deg, color, 8, cv2.LINE_AA)
    cv2.ellipse(out, (cx, cy), (radius, radius), -90, 0, sweep_deg, color, 3, cv2.LINE_AA)

    _draw_pill(out, hold.label, (max(12, cx - 44), max(60, cy - radius - 12)), color)


def _draw_flash_banner(out: np.ndarray, text: str, alpha: float, w: int, h: int) -> None:
    """High-contrast fade-out confirmation banner when a discrete gesture fires."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cx = w // 2
    y = 96
    x1 = max(12, cx - tw // 2 - 22)
    y1 = y - th - 14
    x2 = min(w - 12, cx + tw // 2 + 22)
    y2 = y + baseline + 12

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), BG_PANEL, -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), ACCENT_GREEN, 3, cv2.LINE_AA)
    cv2.putText(overlay, text, (cx - tw // 2, y), font, scale, TEXT_PRIMARY, thickness + 1, cv2.LINE_AA)
    cv2.putText(overlay, text, (cx - tw // 2, y), font, scale, ACCENT_GREEN, thickness, cv2.LINE_AA)
    blend = min(0.95, 0.35 + 0.65 * alpha)
    cv2.addWeighted(overlay, blend, out, 1.0 - blend, 0, dst=out)


def _draw_debug_card(out: np.ndarray, event: GestureEvent, h: int) -> None:
    """Compact corner telemetry box showing live normalized gesture math."""
    dbg = event.debug
    lines = [
        f"HAND:  {dbg.handedness}",
        f"PALM:  {dbg.palm_size:.3f}",
        f"PINCH: {dbg.pinch_ratio:.2f}",
        f"FING:  {dbg.fingers}",
        f"VX:    {dbg.vx:+.2f}",
    ]
    x, y = 12, h - 140
    overlay = out.copy()
    cv2.rectangle(overlay, (x, y), (x + 150, y + 98), BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0, dst=out)
    cv2.rectangle(out, (x, y), (x + 150, y + 98), ACCENT_CYAN, 1, cv2.LINE_AA)

    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (x + 8, y + 18 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )


def _draw_pill(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    accent_color: tuple[int, int, int],
) -> None:
    """Draw a compact dark label pill with an accent border."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.44
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    pad_x, pad_y = 7, 4
    cv2.rectangle(
        image,
        (x - pad_x, y - th - pad_y),
        (x + tw + pad_x, y + baseline + pad_y - 2),
        BG_PANEL,
        -1,
    )
    cv2.rectangle(
        image,
        (x - pad_x, y - th - pad_y),
        (x + tw + pad_x, y + baseline + pad_y - 2),
        accent_color,
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(image, text, (x, y), font, scale, accent_color, thickness, cv2.LINE_AA)
