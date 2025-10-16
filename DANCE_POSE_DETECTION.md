# 舞蹈视频人体姿态检测说明

本文档介绍如何使用YOLOv8-pose模型检测舞蹈视频中的人体关键点，包括头部、肩膀、肘部、腕部等大关节位置。

## 功能特点

- 检测视频中人体的17个关键点
- 实时绘制骨骼连接线
- 支持批量处理dance目录下的所有视频
- 彩色标记不同类型的关键点：
  - 头部关键点：红色
  - 上肢关键点：蓝色
  - 下肢关键点：黄色
- 显示关键点标签名称

## 检测的关键点

1. nose (鼻子)
2. left_eye (左眼)
3. right_eye (右眼)
4. left_ear (左耳)
5. right_ear (右耳)
6. left_shoulder (左肩)
7. right_shoulder (右肩)
8. left_elbow (左肘)
9. right_elbow (右肘)
10. left_wrist (左腕)
11. right_wrist (右腕)
12. left_hip (左髋)
13. right_hip (右髋)
14. left_knee (左膝)
15. right_knee (右膝)
16. left_ankle (左踝)
17. right_ankle (右踝)

## 使用方法

### 1. 环境准备

确保已安装必要的依赖包：

```bash
pip install -r requirements.txt
```

### 2. 运行姿态检测脚本

```bash
python src/final_dance_pose_detector.py
```

### 3. 选择处理模式

脚本提供三种处理模式：

1. **处理单个视频文件**：选择dance目录下的单个视频进行处理
2. **批量处理所有视频**：自动处理dance目录下的所有视频文件
3. **摄像头实时处理**：使用摄像头进行实时姿态检测

### 4. 查看结果

处理后的视频将保存在 `runs/final_dance_pose/` 目录下，文件名格式为：
`{原文件名}_pose_result.mp4`

## 代码结构

### 主要文件

- `src/final_dance_pose_detector.py`：主要的姿态检测脚本
- `src/pose_detection_dance.py`：基础版本的姿态检测脚本
- `src/batch_pose_detection_dance.py`：批量处理脚本

### 核心功能

1. **DancePoseDetector类**：
   - 初始化YOLOv8-pose模型
   - 处理视频帧
   - 绘制关键点和骨骼连接

2. **关键点绘制**：
   - `draw_skeleton()`：绘制骨骼连接线和关键点
   - 不同身体部位使用不同颜色标记
   - 显示关键点名称标签

3. **视频处理**：
   - `process_video()`：处理单个视频文件
   - `process_all_dance_videos()`：批量处理所有视频

## 技术细节

### 模型选择

使用YOLOv8-pose模型进行姿态估计：
- 模型文件：`yolo11n-pose.pt`
- 设备：CPU（避免MPS相关问题）
- 置信度阈值：0.3

### 坐标处理

- 直接使用模型输出的像素坐标
- 避免重复的坐标转换计算
- 提高处理效率

### 性能优化

- 使用`torch.no_grad()`禁用梯度计算
- 强制使用CPU设备避免MPS问题
- 批量处理减少模型加载次数

## 常见问题

### 1. MPS相关警告

由于Apple MPS在姿态估计模式下存在已知问题，脚本强制使用CPU设备。

### 2. 模型下载

首次运行时会自动从Ultralytics下载预训练模型。

### 3. 视频格式支持

支持常见的视频格式：.mp4, .avi, .mov, .mkv

## 扩展功能

### 自定义置信度阈值

可以调整`confidence_threshold`参数来控制检测精度：
- 较高值（如0.7）：更精确但可能漏检
- 较低值（如0.2）：更多检测但可能误检

### 处理其他视频目录

修改`dance_dir`参数可以处理其他目录的视频文件。

## 输出示例

处理后的视频将显示：
- 彩色关键点标记
- 绿色骨骼连接线
- 关键点名称标签
- 帧数和检测人数信息

## 性能指标

典型处理速度：
- 分辨率：1920x1080
- FPS：~18帧/秒
- 处理时间：约5秒（96帧视频）

## 注意事项

1. 确保dance目录下有视频文件
2. 首次运行需要网络连接下载模型
3. 处理高分辨率视频可能需要较长时间
4. 输出文件较大，确保有足够的磁盘空间