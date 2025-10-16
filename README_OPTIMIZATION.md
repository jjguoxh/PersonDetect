# Apple Silicon M2 GPU优化说明

本项目针对Apple Silicon M2处理器进行了GPU优化，以提升YOLO目标检测的运行速度。

## 优化内容

### 1. MPS (Metal Performance Shaders) 支持
- 利用Apple Silicon的GPU加速PyTorch计算
- 自动检测MPS可用性并切换到GPU设备
- 避免了半精度计算可能导致的兼容性问题

### 2. 性能优化技术
- 使用`torch.no_grad()`禁用梯度计算
- 模型预加载和设备迁移
- 优化的视频处理流程
- 减少不必要的数据复制

## 文件说明

### [detect_with_ellipses.py](file:///Users/guoxh/Documents/project/yoloset/src/detect_with_ellipses.py)
- 原始检测脚本的优化版本
- 添加了MPS支持
- 保留了原有的所有功能

### [optimized_detect.py](file:///Users/guoxh/Documents/project/yoloset/src/optimized_detect.py)
- 完全重构的优化版本（使用大模型）
- 面向对象设计
- 更好的性能监控和错误处理
- 详细的处理进度和性能统计

### [optimized_detect_small.py](file:///Users/guoxh/Documents/project/yoloset/src/optimized_detect_small.py)
- 完全重构的优化版本（使用小模型）
- 面向对象设计
- 更好的性能监控和错误处理
- 使用更小的YOLOv11n模型，显著提升性能
- 支持批量处理video目录下所有MP4文件

### [batch_process_videos.py](file:///Users/guoxh/Documents/project/yoloset/src/batch_process_videos.py)
- 专门用于批量处理video目录下所有MP4文件的脚本
- 自动处理目录中的所有视频文件
- 生成带椭圆形标记的检测结果
- **支持多对象检测**：人、汽车、卡车、公交车、摩托车

### [test_multi_object_detection.py](file:///Users/guoxh/Documents/project/yoloset/src/test_multi_object_detection.py)
- 测试多对象检测功能的脚本
- 验证是否能正确检测人、汽车、卡车、公交车、摩托车等对象

### [benchmark_test.py](file:///Users/guoxh/Documents/project/yoloset/src/benchmark_test.py)
- 专门的性能基准测试脚本
- 准确测量CPU和MPS版本的性能差异

### [test_smaller_model.py](file:///Users/guoxh/Documents/project/yoloset/src/test_smaller_model.py)
- 测试大模型和小模型性能差异的脚本

### [accuracy_comparison.py](file:///Users/guoxh/Documents/project/yoloset/src/accuracy_comparison.py)
- 准确率对比测试脚本
- 比较大模型和小模型的检测准确率

### [accuracy_comparison_threshold.py](file:///Users/guoxh/Documents/project/yoloset/src/accuracy_comparison_threshold.py)
- 不同置信度阈值下的准确率对比测试脚本

### [comprehensive_test.py](file:///Users/guoxh/Documents/project/yoloset/src/comprehensive_test.py)
- 综合测试脚本
- 同时测试性能和准确率，并生成详细报告

### [export_to_coreml.py](file:///Users/guoxh/Documents/project/yoloset/src/export_to_coreml.py)
- 将YOLO模型导出为CoreML格式的脚本

## 使用方法

### 运行优化检测脚本
```bash
# 使用大模型优化版本处理视频文件（高精度）
python src/optimized_detect.py

# 使用小模型优化版本处理视频文件（高性能）
python src/optimized_detect_small.py

# 选择选项1处理单个视频文件
# 选择选项2使用摄像头实时处理
# 选择选项3批量处理video目录下所有MP4文件

# 批量处理video目录下所有MP4文件（推荐）
python src/batch_process_videos.py

# 测试多对象检测功能
python src/test_multi_object_detection.py
```

### 运行性能测试
```bash
# 进行准确的性能基准测试
python src/benchmark_test.py

# 比较大模型和小模型的性能
python src/test_smaller_model.py

# 比较大模型和小模型的准确率
python src/accuracy_comparison.py

# 测试不同置信度阈值下的准确率
python src/accuracy_comparison_threshold.py

# 综合性能和准确率测试
python src/comprehensive_test.py
```

### 导出CoreML模型
```bash
# 导出小模型为CoreML格式（推荐用于iOS应用）
python src/export_to_coreml.py --weights model/yolo11n.pt --imgsz 640 --half --nms
```

## 检测类别

当前版本支持检测以下对象类别：
- **人** (person) - 黄色椭圆标记
- **汽车** (car) - 青色椭圆标记
- **卡车** (truck) - 品红色椭圆标记
- **公交车** (bus) - 橙色椭圆标记
- **摩托车** (motorcycle) - 紫色椭圆标记

## 综合测试结果

在Apple Silicon M2处理器上进行全面测试：

### 性能结果
| 模型 | 设备 | FPS | 平均每帧时间 | 相对于大模型CPU提升 |
|------|------|-----|-------------|-------------------|
| YOLOv11x (大模型) | CPU | 2.95 | 0.3390秒 | 基准 |
| YOLOv11x (大模型) | MPS | 3.89 | 0.2572秒 | 31.83% |
| YOLOv11n (小模型) | CPU | 23.61 | 0.0424秒 | 700.24% |
| YOLOv11n (小模型) | MPS | 23.14 | 0.0432秒 | 495.14% |

### 准确率分析（置信度0.3）
- 大模型平均每帧检测：23.56个对象
- 小模型平均每帧检测：4.06个对象
- 精度损失：82.77%

### 不同置信度阈值下的检测结果对比
| 置信度阈值 | 大模型检测数 | 小模型检测数 | 差异百分比 |
|-----------|-------------|-------------|-----------|
| 0.1 | 3646 | 1459 | 59.98% |
| 0.2 | 2246 | 576 | 74.35% |
| 0.3 | 1178 | 203 | 82.77% |
| 0.4 | 621 | 58 | 90.66% |
| 0.5 | 347 | 41 | 88.18% |

### 模型大小比较
| 模型 | 文件大小 | 相对于大模型减少 |
|------|---------|----------------|
| YOLOv11x (大模型) | 109.33MB | 基准 |
| YOLOv11n (小模型) | 5.35MB | 95.10% 减少 |

### 实际视频处理测试
使用[optimized_detect_small.py](file:///Users/guoxh/Documents/project/yoloset/src/optimized_detect_small.py)处理一个474帧的视频：
- 处理速度: 11.18 FPS
- 总耗时: 42.40 秒
- 相比大模型MPS版本提升: 近3倍性能

批量处理测试（5个视频文件）：
- 平均处理速度: 22.25 FPS
- 总处理时间: 137.13 秒
- 成功处理: 5/5 个视频文件

## 系统要求

- macOS 12.0或更高版本
- Python 3.8或更高版本
- PyTorch 2.0或更高版本，支持MPS后端
- Apple Silicon M1/M2/M3处理器

## 依赖安装

```bash
pip install -r requirements.txt
```

## 优化建议

1. **根据应用场景选择合适的模型**：
   - 精度优先：使用大模型(YOLOv11x)，置信度0.3-0.5
   - 性能优先：使用小模型(YOLOv11n)，置信度0.1-0.2
   - 平衡考虑：使用小模型(YOLOv11n)，置信度0.2-0.3

2. **使用MPS加速**：相比纯CPU处理，MPS可提供额外31%的性能提升

3. **导出CoreML模型**：对于iOS应用，将模型导出为CoreML格式可获得最佳性能

4. **批量处理**：使用[batch_process_videos.py](file:///Users/guoxh/Documents/project/yoloset/src/batch_process_videos.py)可以自动处理video目录下的所有MP4文件

5. **多对象检测**：现在支持检测人、汽车、卡车、公交车、摩托车等多种对象

6. **进一步优化**：参考[PERFORMANCE_OPTIMIZATION_GUIDE.md](file:///Users/guoxh/Documents/project/yoloset/PERFORMANCE_OPTIMIZATION_GUIDE.md)获取更多优化建议

## 注意事项

1. 首次运行时，PyTorch会自动下载MPS相关组件

2. 如果遇到兼容性问题，可以手动设置设备为CPU

3. 我们移除了可能导致问题的半精度计算，确保稳定运行

4. 在某些情况下，如果MPS出现兼容性问题，程序会自动回退到CPU处理

5. 对于大型视频文件，使用小模型+MPS优化可以显著减少总处理时间

6. **重要**：小模型虽然性能显著提升（495%），但会带来82.77%的精度损失，需要根据实际应用需求权衡选择

7. **建议**：如果精度对您的应用至关重要，请使用大模型；如果性能更重要且可以接受一定的精度损失，请使用小模型

8. **多对象检测**：程序现在支持检测多种对象类别，不仅仅是人