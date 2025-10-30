#!/usr/bin/env python3
"""
Export a YOLO .pt model to CoreML (.mlpackage or .mlmodel) for iOS apps.

Usage:
  python3 src/export_to_coreml.py --weights src/model/yolo11n.pt --imgsz 640 --half --nms

Requirements:
  pip install ultralytics coremltools onnx

Notes:
  - Uses Ultralytics export pipeline for YOLOv8/YOLO11 .pt weights.
  - If local weights are missing, tries to auto-download official yolo11n.pt.
  - Generates .mlpackage under runs/models/export when possible.
  - If .mlpackage export fails (e.g., BlobWriter not loaded), falls back to ONNX→CoreML and saves .mlmodel.
"""

import argparse
import sys
import os
import glob
from pathlib import Path
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO .pt to CoreML .mlpackage")
    parser.add_argument("--weights", type=str, default="src/model/yolo11n.pt", help="Path to YOLO .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export")
    parser.add_argument("--half", action="store_true", help="Use FP16 for export (smaller, faster)")
    parser.add_argument("--nms", action="store_true", help="Include NMS in exported CoreML model")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic input shapes if supported")
    parser.add_argument("--fallback-onnx", action="store_true", help="Force ONNX→CoreML fallback path")
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


def _ensure_weights(weights: Path) -> Path:
    """Ensure local weights exist; try auto-download via Ultralytics if missing."""
    from ultralytics import YOLO  # type: ignore
    if weights.exists():
        return weights
    print(f"[WARN] Weights not found locally: {weights}")
    print("[INFO] Attempting to auto-download official yolo11n.pt via Ultralytics...")
    try:
        _ = YOLO("yolo11n.pt")  # triggers auto-download
        # try to locate the downloaded file and copy into target
        dl_candidates = [
            Path.cwd(),
            Path.home() / ".cache" / "ultralytics",
        ]
        found = None
        for d in dl_candidates:
            try:
                for p in glob.glob(str(d / "**" / "yolo11n.pt"), recursive=True):
                    found = Path(p)
                    break
                if found:
                    break
            except Exception:
                pass
        if found:
            weights.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found, weights)
            print(f"[INFO] Downloaded weights copied to: {weights}")
            return weights
    except Exception as e:
        print(f"[ERROR] Auto-download failed: {e}")
    print("[ERROR] Could not obtain weights.")
    sys.exit(1)


def export_coreml(weights: str, imgsz: int, half: bool, nms: bool, dynamic: bool, force_fallback: bool = False) -> Path:
    from ultralytics import YOLO  # type: ignore
    import coremltools as ct  # type: ignore

    wpath = _ensure_weights(Path(weights))
    print(f"[INFO] Loading model from: {wpath}")
    try:
        model = YOLO(str(wpath))
    except Exception as e:
        print(f"[WARN] Failed to load local weights at {wpath}: {e}")
        print("[INFO] Retrying by loading official identifier 'yolo11n.pt' via Ultralytics...")
        model = YOLO("yolo11n.pt")

    if not force_fallback:
        print("[INFO] Starting CoreML .mlpackage export (Ultralytics)...")
        try:
            out = model.export(
                format="coreml",
                imgsz=imgsz,
                half=half,
                nms=nms,
                dynamic=dynamic,
            )
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
                latest = max(candidates, key=lambda p: os.path.getmtime(p))
                mlpackage = Path(latest)
                print(f"[SUCCESS] CoreML model generated: {mlpackage}")
                return mlpackage
            if isinstance(out, (str, Path)) and str(out).endswith(".mlpackage"):
                mlpackage = Path(out)
                print(f"[SUCCESS] CoreML model generated: {mlpackage}")
                return mlpackage
            print("[WARN] .mlpackage not found after Ultralytics export. Will try ONNX→CoreML fallback.")
        except Exception as e:
            print(f"[WARN] Ultralytics .mlpackage export failed: {e}")
            print("[INFO] Falling back to ONNX→CoreML .mlmodel path...")

    # Fallback path: convert from PyTorch TraceScript to CoreML .mlmodel (neuralnetwork)
    import torch  # type: ignore
    print("[INFO] Exporting via PyTorch TraceScript → CoreML .mlmodel (neuralnetwork)...")
    try:
        model.model.eval()
        example = torch.randn(1, 3, imgsz, imgsz)
        with torch.no_grad():
            ts = torch.jit.trace(model.model, example)
        mlmodel = ct.convert(
            ts,
            source="pytorch",
            inputs=[ct.TensorType(shape=example.shape)],
            compute_units=ct.ComputeUnit.ALL,
            convert_to="neuralnetwork",
        )
        mlmodel_path = Path.cwd() / f"{wpath.stem}.mlmodel"
        mlmodel.save(str(mlmodel_path))
        print(f"[SUCCESS] CoreML .mlmodel generated: {mlmodel_path}")
        return mlmodel_path
    except Exception as e:
        print(f"[ERROR] PyTorch TraceScript → CoreML conversion failed: {e}")
        sys.exit(4)


def main():
    args = parse_args()
    check_requirements()
    mlpackage = export_coreml(
        weights=args.weights,
        imgsz=args.imgsz,
        half=args.half,
        nms=args.nms,
        dynamic=args.dynamic,
        force_fallback=args.fallback_onnx,
    )
    # Copy to iOS Models directory with canonical name
    target_dir = Path("ios/RealTimeDetectApp/Models")
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.weights).stem
    # Decide target name and source path type
    src_path = Path(mlpackage)
    if src_path.suffix == ".mlmodel":
        target_path = target_dir / f"{stem}.mlmodel"
        shutil.copy2(src_path, target_path)
        print(f"[NEXT] Copied .mlmodel to iOS Models: {target_path}")
        print("      Xcode will compile it to .mlmodelc automatically.")
    else:
        # .mlpackage directory
        target_path = target_dir / f"{stem}.mlpackage"
        shutil.copytree(src_path, target_path, dirs_exist_ok=True)
        print(f"[NEXT] Copied .mlpackage to iOS Models: {target_path}")
    print("      Use Vision (VNCoreMLRequest) or CoreML APIs to run inference.")


if __name__ == "__main__":
    main()