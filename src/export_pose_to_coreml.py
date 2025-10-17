#!/usr/bin/env python3
"""
将 YOLO 姿态估计权重 (.pt) 导出为 CoreML (.mlpackage)

用法示例：
  python3 src/export_pose_to_coreml.py --weights model/yolon-pose.pt --imgsz 640 --half --nms

依赖：
  pip install ultralytics coremltools onnx

说明：
  - 使用 Ultralytics 的内置导出能力，支持 YOLOv8/YOLO11 的 pose 权重。
  - 若指定的本地权重不存在，将自动尝试下载官方的 yolo11n-pose.pt 并继续导出。
  - 默认在 runs/models/export 下生成 .mlpackage，同时会复制一份到 macos/model 目录。
"""

import argparse
import sys
import os
import glob
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO Pose .pt to CoreML .mlpackage")
    parser.add_argument("--weights", type=str, default="model/yolon-pose.pt", help="Path to YOLO pose .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export")
    parser.add_argument("--half", action="store_true", help="Use FP16 for export (smaller, faster)")
    parser.add_argument("--nms", action="store_true", help="Include NMS in exported CoreML model")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic input shapes if supported")
    parser.add_argument("--copy", action="store_true", help="Copy generated .mlpackage to macos/model directory")
    return parser.parse_args()


def check_requirements():
    try:
        import ultralytics  # noqa: F401
    except Exception as e:
        print("[ERROR] 未安装 ultralytics 或导入失败\n        请执行: pip install ultralytics")
        print(f"        详情: {e}")
        sys.exit(1)
    try:
        import coremltools  # noqa: F401
    except Exception as e:
        print("[ERROR] 未安装 coremltools 或导入失败\n        请执行: pip install coremltools")
        print(f"        详情: {e}")
        sys.exit(1)
    try:
        import onnx  # noqa: F401
    except Exception:
        print("[WARN] 未安装 onnx。建议安装以提高导出兼容性: pip install onnx")


def export_coreml(weights: str, imgsz: int, half: bool, nms: bool, dynamic: bool) -> Path:
    from ultralytics import YOLO  # type: ignore

    wpath = Path(weights)
    model = None

    if wpath.exists():
        print(f"[INFO] 使用本地权重: {wpath}")
        model = YOLO(str(wpath))
    else:
        print(f"[WARN] 未找到本地权重: {wpath}")
        print("[INFO] 尝试下载官方权重: yolo11n-pose.pt")
        try:
            model = YOLO("yolo11n-pose.pt")  # Ultralytics 将自动下载
            # 尝试将下载的权重拷贝到指定路径（如果存在）
            dl_candidates = [
                Path.cwd(),
                Path.home() / ".cache" / "ultralytics",
            ]
            found_pt = None
            for d in dl_candidates:
                try:
                    for p in glob.glob(str(d / "**" / "yolo11n-pose.pt"), recursive=True):
                        found_pt = Path(p)
                        break
                    if found_pt:
                        break
                except Exception:
                    pass
            if found_pt:
                wpath.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(found_pt, wpath)
                    print(f"[INFO] 已将下载的权重复制到: {wpath}")
                except Exception as e:
                    print(f"[WARN] 复制下载权重到目标路径失败: {e}")
        except Exception as e:
            print(f"[ERROR] 下载 yolo11n-pose.pt 失败: {e}")
            sys.exit(1)

    print("[INFO] 开始导出 CoreML...")
    out = model.export(
        format="coreml",
        imgsz=imgsz,
        half=half,
        nms=nms,
        dynamic=dynamic,
    )

    print("[INFO] 搜索生成的 .mlpackage...")
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
        print(f"[SUCCESS] 已生成 CoreML 模型: {mlpackage}")
        return mlpackage

    if isinstance(out, (str, Path)) and str(out).endswith(".mlpackage"):
        mlpackage = Path(out)
        print(f"[SUCCESS] 已生成 CoreML 模型: {mlpackage}")
        return mlpackage

    print("[ERROR] 导出完成但未找到 .mlpackage 文件。请检查 runs/models/export 目录。")
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

    target_dir = Path("macos/model")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_name = Path(args.weights).stem + ".mlpackage"
    target_path = target_dir / target_name

    try:
        shutil.copytree(mlpackage, target_path, dirs_exist_ok=True)
        print(f"[NEXT] 已复制到: {target_path}")
        print("      可在 macOS 项目的 build_and_run.sh 中通过 --model 指向该路径")
    except Exception as e:
        print(f"[WARN] 复制到 macos/model 目录失败: {e}")

    print("\n[DONE] CoreML 导出流程完成。")


if __name__ == "__main__":
    main()