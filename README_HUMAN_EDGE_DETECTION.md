# 人体边缘检测工具使用说明

这个工具使用YOLOv8的实例分割功能来检测图片中的人体边缘并进行标记。

## 功能特点

- 使用YOLOv8实例分割模型检测人体
- 标记人体的不规则边缘轮廓
- 支持单张图片和批量处理
- 支持Apple Silicon GPU加速
- **自动检测并下载模型文件**
- **优化的头发边缘处理**

## 头发边缘优化特性

本工具特别优化了头发等细节部分的边缘处理：

1. **多级边缘检测**：使用多种边缘检测算法组合，提高细节识别精度
2. **双边滤波**：在保持边缘清晰的同时减少噪声
3. **自适应阈值分割**：根据不同区域自动调整分割阈值
4. **距离变换**：增强边缘连续性，特别适用于头发丝等细小结构
5. **多尺度轮廓绘制**：绘制不同粗细的轮廓以突出细节

## 快速开始

### 1. 测试脚本

可以使用测试脚本来验证安装和功能是否正常：

```bash
python src/test_human_edge_detector.py
```

该脚本会自动：
- 下载示例图片
- 下载所需的分割模型
- 运行人体边缘检测
- 生成测试结果

## 使用方法

### 1. 自动模型下载

程序会自动检查模型文件是否存在，如果不存在则自动下载:

- 检查 `model/yolo11x-seg.pt` 是否存在
- 如果不存在，程序会自动下载模型文件
- 无需手动下载模型

### 2. 单张图片处理

```bash
python src/human_edge_detector.py --input path/to/your/image.jpg --output result/output.jpg
```

### 3. 批量处理图片

```bash
python src/human_edge_detector.py --input path/to/input/directory --output path/to/output/directory
```

### 4. 参数说明

- `--input` 或 `-i`: 输入图片路径或目录（必需）
- `--output` 或 `-o`: 输出图片路径或目录
- `--model` 或 `-m`: YOLO分割模型路径（默认: model/yolo11x-seg.pt）
- `--confidence` 或 `-c`: 置信度阈值（默认: 0.5）
- `--no-gpu`: 禁用GPU加速，使用CPU处理

### 5. 示例

```bash
# 处理单张图片（自动下载模型）
python src/human_edge_detector.py -i sample.jpg -o result.jpg

# 批量处理图片
python src/human_edge_detector.py -i input_images/ -o output_images/

# 使用自定义模型和置信度
python src/human_edge_detector.py -i sample.jpg -o result.jpg -m my_model.pt -c 0.7

# 在CPU上运行（不使用GPU加速）
python src/human_edge_detector.py -i sample.jpg -o result.jpg --no-gpu
```

## 输出效果

程序会在检测到的人体边缘绘制彩色轮廓，并以半透明的方式显示在原图上，清晰地标记出人体的不规则边缘，特别优化了头发等细节部分的显示效果。

## 注意事项

1. 首次运行时会自动下载模型文件，请确保网络连接正常
2. 分割模型文件较大，下载可能需要一些时间
3. 在Apple Silicon设备上会自动使用MPS加速，提高处理速度
4. 如果遇到兼容性问题，可以使用`--no-gpu`参数禁用GPU加速
5. 模型文件会自动保存到指定路径，下次运行时直接使用