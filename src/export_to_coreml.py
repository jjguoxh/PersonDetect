#!/usr/bin/env python3
"""
Export a YOLO .pt model to CoreML (.mlpackage) for iOS apps.

Usage:
  python3 export_to_coreml.py --weights model/yolo11x.pt --imgsz 640 --half --nms

Requirements:
  pip install ultralytics coremltools onnx

Notes:
  - This script uses Ultralytics export pipeline, which supports YOLOv8/YOLOv11 .pt weights.
  - The output .mlpackage will be created under a "runs/models/export" directory by default.
"""

import argparse
import sys
import os
import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO .pt to CoreML .mlpackage")
    parser.add_argument("--weights", type=str, default="model/yolo11x.pt", help="Path to YOLO .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export")
    parser.add_argument("--half", action="store_true", help="Use FP16 for export (smaller, faster)")
    parser.add_argument("--nms", action="store_true", help="Include NMS in exported CoreML model")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic input shapes if supported")
    return parser.parse_args()


def check_requirements():
    try:
        import ultralytics  # noqa: F401
    except Exception as e:
        print("[ERROR] Ultralytics not installed or failed to import.")
        print("        Install with: pip install ultralytics")
        print(f"        Details: {e}")
        sys.exit(1)
    try:
        import coremltools  # noqa: F401
    except Exception as e:
        print("[ERROR] coremltools not installed or failed to import.")
        print("        Install with: pip install coremltools")
        print(f"        Details: {e}")
        sys.exit(1)
    # ONNX is sometimes required as intermediate for export
    try:
        import onnx  # noqa: F401
    except Exception:
        print("[WARN] onnx not installed. Installing ONNX is recommended: pip install onnx")


def export_coreml(weights: str, imgsz: int, half: bool, nms: bool, dynamic: bool) -> Path:
    # 修复导入问题
    from ultralytics import YOLO  # type: ignore

    # Validate weights path
    wpath = Path(weights)
    if not wpath.exists():
        print(f"[ERROR] Weights file not found: {wpath}")
        sys.exit(1)

    print(f"[INFO] Loading model from: {wpath}")
    model = YOLO(str(wpath))

    print("[INFO] Starting CoreML export...")
    # Ultralytics export parameters
    # Reference: model.export(format='coreml', imgsz=imgsz, half=half, nms=nms, dynamic=dynamic)
    out = model.export(
        format="coreml",
        imgsz=imgsz,
        half=half,
        nms=nms,
        dynamic=dynamic,
    )

    # Try to locate generated .mlpackage
    print("[INFO] Searching for generated .mlpackage...")
    search_dirs = [
        Path.cwd(),
        Path.cwd() / "runs" / "models" / "export",
        wpath.parent,
    ]
    candidates = []
    for d in search_dirs:
        try:
            candidates.extend(glob.glob(str(d / "**" / "*.mlpackage"), recursive=True))
        except Exception:
            pass

    if candidates:
        # Prefer the most recently modified
        latest = max(candidates, key=lambda p: os.path.getmtime(p))
        mlpackage = Path(latest)
        print(f"[SUCCESS] CoreML model generated: {mlpackage}")
        return mlpackage

    # If Ultralytics returned a path, try to use it
    if isinstance(out, (str, Path)) and str(out).endswith(".mlpackage"):
        mlpackage = Path(out)
        print(f"[SUCCESS] CoreML model generated: {mlpackage}")
        return mlpackage

    print("[ERROR] Export completed but .mlpackage file was not found.")
    print("        Please check the output folders (e.g., runs/models/export).")
    sys.exit(2)


def main():
    args = parse_args()
    check_requirements()
    mlpackage = export_coreml(
        weights=args.weights,
        imgsz=args.imgsz,
        half=args.half,
        nms=args.nms,
        dynamic=args.dynamic,
    )
    print("\n[NEXT] Add the .mlpackage to your Xcode project.")
    print("      Xcode will compile it to .mlmodelc automatically.")
    print("      Use Vision (VNCoreMLRequest) or CoreML APIs to run inference.")


if __name__ == "__main__":
    main()