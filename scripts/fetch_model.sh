#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/models"
MODEL_PATH="$MODEL_DIR/hand_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_PATH" ]]; then
  echo "Model already exists at: $MODEL_PATH"
  exit 0
fi

echo "Downloading MediaPipe Hand Landmarker model (~7.5 MB)..."
curl -fsSL "$MODEL_URL" -o "$MODEL_PATH"
echo "Saved to: $MODEL_PATH"
