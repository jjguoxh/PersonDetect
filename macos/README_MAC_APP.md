# EllipsesDetect (macOS CLI)

一个在 macOS 上运行的命令行程序，使用 CoreML + Vision 对视频中的人物进行检测，并在人物底部叠加椭圆标记，功能对齐 `src/detect_with_ellipses.py`。

## 前置要求
- macOS（需安装 Xcode 工具链，提供 `xcrun`/`swiftc`）
- 将转换好的 CoreML 模型（`.mlpackage`）放入 `macos/model/yolo11x.mlpackage`

## 构建与运行

```bash
# 构建并运行（默认使用 video/market-square.mp4）
bash macos/build_and_run.sh

# 自定义参数
MODEL_PATH=macos/model/yolo11x.mlpackage \
INPUT_PATH=video/football.mp4 \
OUTPUT_PATH=result/football_with_ellipses.mp4 \
CONFIDENCE=0.3 \
bash macos/build_and_run.sh
```

或直接使用可执行文件：

```bash
xcrun swiftc macos/EllipsesDetectCLI.swift -o macos/ellipses_cli \
  -framework CoreML -framework Vision -framework AVFoundation \
  -framework CoreImage -framework CoreGraphics

./macos/ellipses_cli --model macos/model/yolo11x.mlpackage \
  --input video/market-square.mp4 \
  --output result/result_with_ellipses.mp4 \
  --confidence 0.3
```

## 说明
- 只处理 `person` 类（标签名为 `person` 或类索引 `0`）。
- 输出视频编码为 H.264（`.mp4`）。
- 若模型导出未包含 NMS，Vision 结果可能不包含 `VNRecognizedObjectObservation`，需要调整解析逻辑。
- 如需 GUI 版（SwiftUI App），可以在此基础上扩展：添加文件选择界面、进度展示等。

## 常见问题
- `macos/model` 为空：请将已转换的 `.mlpackage` 拷贝到该目录，文件名为 `Model.mlpackage` 或在运行时通过 `--model` 指定路径。
- 性能：对高分辨率视频逐帧处理较慢，可在模型侧启用 NMS、降低分辨率或提高置信度阈值。