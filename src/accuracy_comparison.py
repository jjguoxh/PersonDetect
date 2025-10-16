#!/usr/bin/env python3
"""
准确率对比测试脚本
比较大模型和小模型在相同测试数据上的检测准确率
"""

import cv2
import torch
# 修复导入问题
from ultralytics import YOLO  # type: ignore
import os
import numpy as np

def count_detections(model, video_path, device='cpu', num_frames=100):
    """
    统计模型检测到的对象数量
    
    Args:
        model: YOLO模型
        video_path: 视频路径
        device: 设备
        num_frames: 测试帧数
        
    Returns:
        total_detections: 总检测数
        frame_detections: 每帧检测数列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    total_detections = 0
    frame_detections = []
    
    frame_count = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 使用YOLO模型进行检测
        results = model(frame, verbose=False, device=device, conf=0.3)
        
        detection_count = 0
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # 获取置信度
                    confidence = float(box.conf[0])
                    # 只统计置信度足够高的人类检测
                    if confidence > 0.3 and int(box.cls[0]) == 0:  # 0是'person'类
                        detection_count += 1
        
        total_detections += detection_count
        frame_detections.append(detection_count)
        frame_count += 1
    
    cap.release()
    return total_detections, frame_detections

def compare_accuracy(large_model_path, small_model_path, video_path, num_frames=100):
    """
    比较两个模型的准确率
    
    Args:
        large_model_path: 大模型路径
        small_model_path: 小模型路径
        video_path: 视频路径
        num_frames: 测试帧数
    """
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"测试帧数: {num_frames}")
    
    # 加载大模型
    print(f"加载大模型: {large_model_path}")
    large_model = YOLO(large_model_path)
    large_model.to(device)
    
    # 加载小模型
    print(f"加载小模型: {small_model_path}")
    small_model = YOLO(small_model_path)
    small_model.to(device)
    
    print("=" * 60)
    
    # 测试大模型
    print("测试大模型 (yolo11x)...")
    large_detections, large_frame_detections = count_detections(
        large_model, video_path, device, num_frames)
    
    large_avg_detections = np.mean(large_frame_detections)
    print(f"大模型总检测数: {large_detections}")
    print(f"大模型平均每帧检测数: {large_avg_detections:.2f}")
    
    print("=" * 60)
    
    # 测试小模型
    print("测试小模型 (yolo11n)...")
    small_detections, small_frame_detections = count_detections(
        small_model, video_path, device, num_frames)
    
    small_avg_detections = np.mean(small_frame_detections)
    print(f"小模型总检测数: {small_detections}")
    print(f"小模型平均每帧检测数: {small_avg_detections:.2f}")
    
    print("=" * 60)
    
    # 计算准确率差异
    detection_diff = large_detections - small_detections
    detection_diff_percent = (detection_diff / large_detections * 100) if large_detections > 0 else 0
    
    avg_diff = large_avg_detections - small_avg_detections
    avg_diff_percent = (avg_diff / large_avg_detections * 100) if large_avg_detections > 0 else 0
    
    print("准确率比较结果:")
    print(f"大模型平均每帧检测数: {large_avg_detections:.2f}")
    print(f"小模型平均每帧检测数: {small_avg_detections:.2f}")
    print(f"平均每帧检测数差异: {avg_diff:.2f} ({avg_diff_percent:.2f}%)")
    print(f"总检测数差异: {detection_diff} ({detection_diff_percent:.2f}%)")
    
    if detection_diff > 0:
        print(f"⚠ 小模型检测到的对象数量比大模型少 {detection_diff_percent:.2f}%")
        print("  这意味着在精度上有所损失")
    elif detection_diff < 0:
        print(f"✓ 小模型检测到的对象数量比大模型多 {abs(detection_diff_percent):.2f}%")
    else:
        print("  两个模型检测到的对象数量相同")
    
    return large_avg_detections, small_avg_detections

def find_test_video():
    """查找测试视频文件"""
    possible_names = [
        "video/market-square.mp4",
        "input.mp4",
        "video.mp4",
        "test.mp4"
    ]
    
    for name in possible_names:
        if os.path.exists(name):
            return name
    
    return None

def main():
    # 模型路径
    large_model = "model/yolo11x.pt"  # 大模型
    small_model = "model/yolo11n.pt"  # 小模型
    
    # 检查模型文件是否存在
    if not os.path.exists(large_model):
        print(f"错误: 找不到大模型文件 {large_model}")
        return
        
    if not os.path.exists(small_model):
        print(f"错误: 找不到小模型文件 {small_model}")
        return
    
    video_path = find_test_video()
    if not video_path:
        print("错误: 找不到测试视频文件")
        return
    
    print(f"测试视频: {video_path}")
    print("=" * 60)
    
    # 进行准确率比较
    large_avg, small_avg = compare_accuracy(large_model, small_model, video_path, num_frames=100)
    
    print("\n" + "=" * 60)
    print("总结:")
    print(f"大模型 (yolo11x): 平均每帧检测 {large_avg:.2f} 个对象")
    print(f"小模型 (yolo11n): 平均每帧检测 {small_avg:.2f} 个对象")
    
    if small_avg < large_avg:
        loss_percent = (large_avg - small_avg) / large_avg * 100
        print(f"精度损失: {loss_percent:.2f}%")
        print("\n建议:")
        print("1. 如果精度对您的应用至关重要，请使用大模型")
        print("2. 如果性能更重要且可以接受一定的精度损失，请使用小模型")
        print("3. 可以调整置信度阈值来平衡精度和召回率")
    else:
        print("小模型在该测试中表现相当或更好")

if __name__ == "__main__":
    main()