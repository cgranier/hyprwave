"""HyprWave CLI entry point (Milestones 1 & 2 runnable right away)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from hyprwave.actions import ActionDispatcher, read_system_brightness, read_system_volume
from hyprwave.camera import Camera
from hyprwave.config import AppConfig, DEFAULT_MODEL_PATH
from hyprwave.gestures import GestureRecognizer
from hyprwave.hud import draw_hud
from hyprwave.tracker import HandTracker
from hyprwave.window import ViewFlags, run_hud_window


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        prog="hyprwave",
        description="Touchless hand-gesture HUD and Hyprland controller.",
    )
    parser.add_argument("--device", type=int, default=0, help="V4L2 camera index (default: 0 -> /dev/video0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--fps", type=float, default=20.0, help="Active target FPS cap (default: 20)")
    parser.add_argument(
        "--idle-fps",
        type=float,
        default=4.0,
        help="Low-power FPS when no hand is detected (0 to disable, default: 4)",
    )
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal mirroring")
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Milestone 1 mode: camera + HUD only (skip MediaPipe)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to MediaPipe hand_landmarker.task model",
    )
    parser.add_argument("--show-indices", action="store_true", help="Overlay landmark numbers 0..20")
    parser.add_argument("--debug-math", action="store_true", help="Show live gesture-math telemetry card")
    parser.add_argument("--live", action="store_true", help="Start in LIVE mode (dispatch real Hyprland/audio/brightness controls)")
    args = parser.parse_args(argv)

    return AppConfig(
        device_index=args.device,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        idle_fps=args.idle_fps,
        mirror=not args.no_mirror,
        model_path=args.model,
        enable_tracking=not args.no_tracking,
        show_landmarks_index=args.show_indices,
        show_debug_math=args.debug_math,
        dry_run=not args.live,
    )


def run(cfg: AppConfig) -> int:
    dispatcher = ActionDispatcher(dry_run=cfg.dry_run)

    tracker: HandTracker | None = None
    if cfg.enable_tracking:
        tracker = HandTracker(
            model_path=cfg.model_path,
            num_hands=cfg.num_hands,
            min_detection_confidence=cfg.min_detection_confidence,
            min_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            mirrored=cfg.mirror,
        )
        try:
            tracker.open()
        except FileNotFoundError as exc:
            print(f"[hyprwave] {exc}", file=sys.stderr)
            print("[hyprwave] Tip: run with --no-tracking to test Milestone 1 (camera only).", file=sys.stderr)
            return 1

    try:
        with Camera(
            device_index=cfg.device_index,
            width=cfg.width,
            height=cfg.height,
            target_fps=cfg.target_fps,
            mirror=cfg.mirror,
        ) as cam:
            w, h = cam.actual_resolution
            aspect = (w / h) if (w and h) else (cfg.width / cfg.height)
            init_vol = read_system_volume()
            init_brt = read_system_brightness()
            recognizer = GestureRecognizer(
                aspect=aspect,
                initial_volume=init_vol,
                initial_brightness=init_brt,
            )

            mode_str = "DEMO (press 'l' for LIVE)" if cfg.dry_run else "LIVE"
            print(f"[hyprwave] Opened /dev/video{cfg.device_index} at {w}x{h} (target {cfg.target_fps:.0f} FPS) | Mode: {mode_str}")
            print(f"[hyprwave] Synced system levels: volume={int(init_vol * 100)}%, brightness={int(init_brt * 100)}%")
            print("[hyprwave] Keys: [l] toggle LIVE/DEMO, [i] indices, [d] math debug, [q]/ESC quit.")

            last_hand_time = time.monotonic()

            def render_next_frame(flags: ViewFlags) -> tuple[np.ndarray, float]:
                nonlocal last_hand_time
                # QTimer in HudWindow paces the frame loop, so skip blocking sleep.
                frame, fps = cam.read(pace=False)
                dispatcher.dry_run = flags.dry_run
                now = time.monotonic()

                if tracker is not None:
                    hands, infer_ms = tracker.process(frame)
                    event = recognizer.update(hands, now)
                    dispatcher.dispatch(event, now)
                    status_text = None

                    if hands:
                        last_hand_time = now
                        is_idle = False
                    else:
                        is_idle = (
                            cfg.idle_fps > 0
                            and (now - last_hand_time) >= cfg.idle_after_s
                        )
                    desired_fps = cfg.idle_fps if is_idle else cfg.target_fps
                else:
                    hands = []
                    infer_ms = None
                    event = None
                    is_idle = False
                    desired_fps = cfg.target_fps
                    status_text = "MILESTONE 1: CAMERA ONLY (--no-tracking)"

                hud_frame = draw_hud(
                    frame,
                    hands,
                    fps=fps,
                    infer_ms=infer_ms,
                    event=event,
                    status_text=status_text,
                    show_indices=flags.show_indices,
                    show_debug=flags.show_debug,
                    dry_run=flags.dry_run,
                    is_idle=is_idle,
                )
                return hud_frame, desired_fps

            return run_hud_window(
                title=cfg.window_title,
                width=w or cfg.width,
                height=h or cfg.height,
                target_fps=cfg.target_fps,
                next_frame=render_next_frame,
                initial_flags=ViewFlags(
                    show_indices=cfg.show_landmarks_index,
                    show_debug=cfg.show_debug_math,
                    dry_run=cfg.dry_run,
                ),
            )
    except KeyboardInterrupt:
        return 0
    finally:
        if tracker is not None:
            tracker.close()


def main() -> None:
    cfg = parse_args()
    raise SystemExit(run(cfg))


if __name__ == "__main__":
    main()
