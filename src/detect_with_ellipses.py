#!/usr/bin/env python3
"""
使用YOLO检测人物并在视频中用椭圆形标记替代矩形边界框
类似于FIFA游戏中球员脚部的标记方式

针对Apple Silicon M2 GPU优化版本
"""

import cv2
import numpy as np
import torch
# 修复导入问题
from ultralytics import YOLO  # type: ignore
import os

def draw_ellipse_marker(frame, bbox, color=(255, 255, 0), thickness=4):
    """
    在检测到的人物底部绘制椭圆形标记
    
    Args:
        frame: 视频帧
        bbox: 边界框坐标 [x1, y1, x2, y2]
        color: 椭圆颜色 (B, G, R)
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
    
    # 可选：在椭圆中心绘制小圆点
    cv2.circle(frame, (center_x, center_y), 3, color, -1)

def process_video_with_ellipses(model_path, video_path, output_path, confidence_threshold=0.5):
    """
    处理视频并在检测到的人物位置绘制椭圆形标记
    
    Args:
        model_path: YOLO模型路径
        video_path: 输入视频路径
        output_path: 输出视频路径
        confidence_threshold: 置信度阈值
    """
    print(f"加载模型: {model_path}")
    # 加载模型并设置为MPS设备（如果可用）
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"使用设备: {device}")
    model = YOLO(model_path)
    model.to(device)  # 将模型移动到MPS设备
    
    print(f"打开视频文件: {video_path}")
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频文件是否成功打开
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
    
    print(f"创建输出视频文件: {output_path}")
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 检查视频写入器是否成功初始化
    if not out.isOpened():
        raise ValueError(f"无法创建输出视频文件: {output_path}")
    
    frame_count = 0
    processed_frames = 0
    
    print("开始处理视频帧...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 使用YOLO模型进行检测
        results = model(frame, verbose=False, device=device, conf=confidence_threshold)
        
        # 处理检测结果
        detection_count = 0
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # 获取置信度
                    confidence = float(box.conf[0])
                    
                    # 只处理置信度足够高的人类检测
                    if confidence > confidence_threshold and int(box.cls[0]) == 0:  # 0是'person'类
                        # 获取边界框坐标
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # 绘制椭圆形标记
                        draw_ellipse_marker(frame, bbox, color=(255, 255, 0), thickness=4)
                        detection_count += 1
        
        # 写入处理后的帧
        out.write(frame)
        frame_count += 1
        
        # 显示进度
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧, 当前帧检测到 {detection_count} 个人物")
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"视频处理完成!")
    print(f"处理帧数: {frame_count}")
    print(f"输出文件: {output_path}")
    print(f"输出文件大小: {os.path.getsize(output_path) if os.path.exists(output_path) else '文件不存在'} 字节")

def process_camera_with_ellipses(model_path, output_path, confidence_threshold=0.5):
    """
    使用摄像头实时检测并在检测到的人物位置绘制椭圆形标记
    
    Args:
        model_path: YOLO模型路径
        output_path: 输出视频路径
        confidence_threshold: 置信度阈值
    """
    print(f"加载模型: {model_path}")
    # 加载模型并设置为MPS设备（如果可用）
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"使用设备: {device}")
    model = YOLO(model_path)
    model.to(device)  # 将模型移动到MPS设备
    
    print("打开摄像头...")
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        raise ValueError("无法打开摄像头")
    
    # 获取摄像头属性
    fps = 30  # 假设FPS为30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"摄像头信息: {width}x{height}, FPS: {fps}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"创建输出视频文件: {output_path}")
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 检查视频写入器是否成功初始化
    if not out.isOpened():
        raise ValueError(f"无法创建输出视频文件: {output_path}")
    
    frame_count = 0
    
    print("开始处理摄像头画面，按 'q' 键退出...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 使用YOLO模型进行检测，指定使用MPS设备
        results = model(frame, verbose=False, device=device, conf=confidence_threshold)
        
        # 处理检测结果
        detection_count = 0
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # 获取置信度
                    confidence = float(box.conf[0])
                    
                    # 只处理置信度足够高的人类检测
                    if confidence > confidence_threshold and int(box.cls[0]) == 0:  # 0是'person'类
                        # 获取边界框坐标
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # 绘制椭圆形标记
                        draw_ellipse_marker(frame, bbox, color=(0, 255, 0), thickness=2)
                        detection_count += 1
        
        # 显示处理后的帧
        cv2.imshow('YOLO Detection with Ellipses', frame)
        
        # 写入处理后的帧
        out.write(frame)
        frame_count += 1
        
        # 显示进度
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧, 当前帧检测到 {detection_count} 个人物")
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"摄像头处理完成!")
    print(f"处理帧数: {frame_count}")
    print(f"输出文件: {output_path}")
    print(f"输出文件大小: {os.path.getsize(output_path) if os.path.exists(output_path) else '文件不存在'} 字节")

def find_video_file():
    """查找可用的视频文件"""
    # 常见的视频文件名
    possible_names = [
        "video/market-square.mp4",
        "input.mp4",
        "video.mp4",
        "test.mp4"
    ]
    
    # 检查当前目录
    for name in possible_names:
        if os.path.exists(name):
            return name
    
    # 检查常见目录
    directories = [".", "data", "videos", "samples"]
    for directory in directories:
        if os.path.exists(directory):
            for name in possible_names:
                path = os.path.join(directory, name)
                if os.path.exists(path):
                    return path
    
    return None

def main():
    # 检查必要的文件是否存在
    model_path = "model/yolo11x.pt"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    # 询问用户选择处理模式
    print("请选择处理模式:")
    print("1. 处理视频文件")
    print("2. 使用摄像头实时处理")
    
    choice = input("请输入选项 (1 或 2): ").strip()
    
    if choice == "2":
        # 使用摄像头模式
        output_dir = "result"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "camera_output_with_ellipses.mp4")
        
        print("使用摄像头模式...")
        print(f"模型: {model_path}")
        print(f"输出视频: {output_path}")
        
        try:
            # 处理摄像头
            process_camera_with_ellipses(model_path, output_path, confidence_threshold=0.3)
            print("完成!")
            
            # 验证输出文件
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                print(f"输出文件已成功创建，大小: {size} 字节")
            else:
                print("警告: 输出文件未找到!")
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 处理视频文件模式
        # 查找视频文件
        video_path = find_video_file()
        if not video_path:
            print("错误: 找不到视频文件")
            print("请确保以下文件之一存在于项目目录中:")
            print("  - market-square.mp4")
            print("  - input.mp4")
            print("  - video.mp4")
            print("  - test.mp4")
            print("\n或者选择摄像头模式 (选项 2)")
            return
        
        # 创建输出目录
        output_dir = "runs/detect_ellipses"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "result_with_ellipses.mp4")
        
        print("开始处理视频...")
        print(f"模型: {model_path}")
        print(f"输入视频: {video_path}")
        print(f"输出视频: {output_path}")
        
        try:
            # 处理视频
            process_video_with_ellipses(model_path, video_path, output_path, confidence_threshold=0.3)
            print("完成!")
            
            # 验证输出文件
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                print(f"输出文件已成功创建，大小: {size} 字节")
            else:
                print("警告: 输出文件未找到!")
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()