#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
SRC="$ROOT_DIR/macos/EllipsesDetectCLI.swift"
BIN="$ROOT_DIR/macos/ellipses_cli"

echo "[INFO] Building ellipses_cli"
xcrun swiftc "$SRC" -o "$BIN" \
  -framework CoreML -framework Vision -framework AVFoundation \
  -framework CoreImage -framework CoreGraphics

MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/macos/model/yolo11x.mlpackage}"
INPUT_PATH="${INPUT_PATH:-$ROOT_DIR/video/market-square.mp4}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/result/market-square.mp4}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "[INFO] Running ellipses_cli"
"$BIN" --model "$MODEL_PATH" --input "$INPUT_PATH" --output "$OUTPUT_PATH" --confidence "${CONFIDENCE:-0.3}" "$@"

echo "[DONE] Output at: $OUTPUT_PATH"