#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
SRC="$ROOT_DIR/macos/DancePoseCLI.swift"
BIN="$ROOT_DIR/macos/dance_pose_cli"

echo "[INFO] Building dance_pose_cli"
xcrun swiftc "$SRC" -o "$BIN" \
  -framework CoreML -framework Vision -framework AVFoundation \
  -framework CoreImage -framework CoreGraphics -framework AppKit

MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/macos/model/yolon-pose.mlpackage}"
DANCE_DIR="${DANCE_DIR:-$ROOT_DIR/dance}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/runs/final_dance_pose}"
CONFIDENCE="${CONFIDENCE:-0.3}"
KP_THRESHOLD="${KP_THRESHOLD:-0.3}"

mkdir -p "$OUT_DIR"

echo "[INFO] Running dance_pose_cli"
"$BIN" --model "$MODEL_PATH" --dir "$DANCE_DIR" --outdir "$OUT_DIR" --confidence "$CONFIDENCE" --kp "$KP_THRESHOLD" "$@"

echo "[DONE] Outputs at: $OUT_DIR"