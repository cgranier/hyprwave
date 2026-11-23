#!/usr/bin/env bash
set -euo pipefail

# Toggle HyprWave on/off from a Hyprland keybinding (Super + Shift + H).
# Sends SIGTERM if already running so Qt/OpenCV release /dev/video0 cleanly.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="${XDG_RUNTIME_DIR:-/tmp}/hyprwave.pid"
HYPRWAVE_BIN="$PROJECT_ROOT/.venv/bin/hyprwave"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

# Fallback check in case HyprWave was launched manually from a terminal
if pgrep -f "$HYPRWAVE_BIN" >/dev/null 2>&1; then
  pkill -TERM -f "$HYPRWAVE_BIN" || true
  exit 0
fi

# Launch in LIVE mode directly from the project's Python 3.12 virtualenv
nohup "$HYPRWAVE_BIN" --live >/dev/null 2>&1 &
echo "$!" > "$PID_FILE"
