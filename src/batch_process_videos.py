#!/usr/bin/env python3
"""
批量处理video目录下所有MP4文件的脚本
"""

import cv2
import numpy as np
import torch
# 修复导入问题
from ultralytics import YOLO  # type: ignore
import os
import time
import glob

# COCO数据集的类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 定义要检测的类别和对应的颜色
DETECTION_CLASSES = {
    'person': (255, 255, 0),    # 黄色
    'car': (0, 255, 255),       # 青色
    'truck': (255, 0, 255),     # 品红
    'bus': (0, 165, 255),       # 橙色
    'motorcycle': (128, 0, 128) # 紫色
}

def draw_object_marker(frame, bbox, class_name, color, thickness=4):
    """
    在检测到的对象底部绘制标记
    
    Args:
        frame: 视频帧
        bbox: 边界框坐标 [x1, y1, x2, y2]
        class_name: 类别名称
        color: 颜色 (B, G, R)
        thickness: 线条粗细
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # 计算椭圆参数
    # 椭圆位于检测框的底部中心
    center_x = (x1 + x2) // 2
    center_y = y2  # 底部
    
    # 椭圆的宽度和高度
    width = (x2 - x1) // 2
    height = max(8, (y2 - y1) // 8)  # 固定高度或基于检测框高度
    
    # 绘制椭圆
    cv2.ellipse(frame, 
                (center_x, center_y), 
                (width, height), 
                0, 0, 360,  # 角度参数
                color, thickness)
    
    # 在椭圆中心绘制小圆点
    cv2.circle(frame, (center_x, center_y), 3, color, -1)
    
    # 添加类别标签
    label = f"{class_name}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_frame(model, frame, device, confidence_threshold=0.5):
    """
    处理单个视频帧
    
    Args:
        model: YOLO模型
        frame: 输入视频帧
        device: 设备
        confidence_threshold: 置信度阈值
        
    Returns:
        processed_frame: 处理后的帧
        detection_count: 检测到的对象数量
    """
    # 使用YOLO模型进行检测
    with torch.no_grad():  # 禁用梯度计算以提高性能
        results = model(frame, verbose=False, device=device, conf=confidence_threshold)
    
    detection_count = 0
    processed_frame = frame.copy()
    
    # 处理检测结果
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # 获取置信度
                confidence = float(box.conf[0])
                
                # 获取类别索引
                class_idx = int(box.cls[0])
                
                # 检查是否是我们要检测的类别
                if confidence > confidence_threshold and class_idx < len(COCO_CLASSES):
                    class_name = COCO_CLASSES[class_idx]
                    
                    # 只处理我们感兴趣的类别
                    if class_name in DETECTION_CLASSES:
                        # 获取边界框坐标
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # 获取对应的颜色
                        color = DETECTION_CLASSES[class_name]
                        
                        # 绘制标记
                        draw_object_marker(processed_frame, bbox, class_name, color, thickness=4)
                        detection_count += 1
                    
    return processed_frame, detection_count

def process_video(model, device, video_path, output_path, confidence_threshold=0.5):
    """
    处理整个视频文件
    
    Args:
        model: YOLO模型
        device: 设备
        video_path: 输入视频路径
        output_path: 输出视频路径
        confidence_threshold: 置信度阈值
    """
    print(f"打开视频文件: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
        
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"无法创建输出视频文件: {output_path}")
        
    frame_count = 0
    start_time = time.time()
    
    print("开始处理视频帧...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 处理帧
        processed_frame, detection_count = process_frame(model, frame, device, confidence_threshold)
        
        # 写入处理后的帧
        out.write(processed_frame)
        frame_count += 1
        
        # 显示进度
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps_processed = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"已处理 {frame_count}/{total_frames} 帧, "
                  f"当前帧检测到 {detection_count} 个对象, "
                  f"处理速度: {fps_processed:.2f} FPS")
            
    # 释放资源
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"视频处理完成!")
    print(f"总帧数: {frame_count}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均处理速度: {avg_fps:.2f} FPS")
    print(f"输出文件: {output_path}")
    print(f"输出文件大小: {os.path.getsize(output_path) if os.path.exists(output_path) else '文件不存在'} 字节")

def process_all_videos():
    """
    处理video目录下的所有MP4文件
    """
    # 使用小模型
    model_path = "model/yolo11n.pt"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
        
    # 设置设备
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    
    # 输入和输出目录
    input_dir = "video"
    output_dir = "runs/batch_processed"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有MP4文件
    video_pattern = os.path.join(input_dir, "*.mp4")
    video_files = glob.glob(video_pattern)
    
    if not video_files:
        print(f"在目录 {input_dir} 中未找到MP4文件")
        return
        
    print(f"找到 {len(video_files)} 个视频文件:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(video_file)}")
        
    print("\n开始批量处理...")
    print(f"检测类别: {', '.join(DETECTION_CLASSES.keys())}")
    
    # 处理每个视频文件
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 处理视频: {os.path.basename(video_file)}")
        
        # 生成输出文件路径
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_path = os.path.join(output_dir, f"{video_name}_result.mp4")
        
        try:
            process_video(model, device, video_file, output_path, confidence_threshold=0.3)
            print(f"✓ 完成处理: {os.path.basename(video_file)}")
        except Exception as e:
            print(f"✗ 处理失败: {os.path.basename(video_file)} - {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\n批量处理完成! 共处理 {len(video_files)} 个视频文件")

def main():
    print("批量处理video目录下所有MP4文件...")
    print("检测类别包括: 人、汽车、卡车、公交车、摩托车")
    process_all_videos()

if __name__ == "__main__":
    main()