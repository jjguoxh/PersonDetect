#!/usr/bin/env python3
"""
针对Apple Silicon M2 GPU优化的YOLO人物检测脚本
使用椭圆形标记替代矩形边界框，类似于FIFA游戏中球员脚部的标记方式
"""

import cv2
import numpy as np
import torch
# 修复导入问题
from ultralytics import YOLO  # type: ignore
import os
import time

class OptimizedEllipseDetector:
    def __init__(self, model_path, use_mps=True):
        """
        初始化优化的椭圆检测器
        
        Args:
            model_path: YOLO模型路径
            use_mps: 是否使用MPS加速
        """
        self.device = 'mps' if use_mps and torch.backends.mps.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 注意：在某些情况下，MPS上的半精度可能会导致问题，所以我们不强制使用半精度
        # 如果需要半精度优化，可以在特定条件下启用
            
    def draw_ellipse_marker(self, frame, bbox, color=(255, 255, 0), thickness=4):
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
        
        # 在椭圆中心绘制小圆点
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
    def process_frame(self, frame, confidence_threshold=0.5):
        """
        处理单个视频帧
        
        Args:
            frame: 输入视频帧
            confidence_threshold: 置信度阈值
            
        Returns:
            processed_frame: 处理后的帧
            detection_count: 检测到的对象数量
        """
        # 使用YOLO模型进行检测
        with torch.no_grad():  # 禁用梯度计算以提高性能
            results = self.model(frame, verbose=False, device=self.device, conf=confidence_threshold)
        
        detection_count = 0
        processed_frame = frame.copy()
        
        # 处理检测结果
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
                        self.draw_ellipse_marker(processed_frame, bbox, color=(255, 255, 0), thickness=4)
                        detection_count += 1
                        
        return processed_frame, detection_count
        
    def process_video(self, video_path, output_path, confidence_threshold=0.5):
        """
        处理整个视频文件
        
        Args:
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
            processed_frame, detection_count = self.process_frame(frame, confidence_threshold)
            
            # 写入处理后的帧
            out.write(processed_frame)
            frame_count += 1
            
            # 显示进度
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps_processed = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"已处理 {frame_count}/{total_frames} 帧, "
                      f"当前帧检测到 {detection_count} 个人物, "
                      f"处理速度: {fps_processed:.2f} FPS")
                
                # 如果处理时间过长，可以考虑跳帧以保持实时性
                # if fps_processed < 10:  # 如果处理速度低于10FPS
                #     print("警告: 处理速度较慢，可能需要降低分辨率或模型复杂度")
                
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
        
    def process_camera(self, output_path, confidence_threshold=0.5):
        """
        使用摄像头实时处理
        
        Args:
            output_path: 输出视频路径
            confidence_threshold: 置信度阈值
        """
        print("打开摄像头...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("无法打开摄像头")
            
        # 获取摄像头属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # 假设FPS为30
        
        print(f"摄像头信息: {width}x{height}, FPS: {fps}")
        
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
        
        print("开始处理摄像头画面，按 'q' 键退出...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 处理帧
            processed_frame, detection_count = self.process_frame(frame, confidence_threshold)
            
            # 显示处理后的帧
            cv2.imshow('Optimized YOLO Detection with Ellipses', processed_frame)
            
            # 写入处理后的帧
            out.write(processed_frame)
            frame_count += 1
            
            # 显示进度
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps_processed = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"已处理 {frame_count} 帧, "
                      f"当前帧检测到 {detection_count} 个人物, "
                      f"处理速度: {fps_processed:.2f} FPS")
                      
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"摄像头处理完成!")
        print(f"总帧数: {frame_count}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均处理速度: {avg_fps:.2f} FPS")
        print(f"输出文件: {output_path}")
        print(f"输出文件大小: {os.path.getsize(output_path) if os.path.exists(output_path) else '文件不存在'} 字节")

def find_video_file():
    """查找可用的视频文件"""
    possible_names = [
        "video/3059549-hd_1920_1080_24fps.mp4",
        "input.mp4",
        "video.mp4",
        "test.mp4"
    ]
    
    for name in possible_names:
        if os.path.exists(name):
            return name
            
    directories = [".", "data", "videos", "samples"]
    for directory in directories:
        if os.path.exists(directory):
            for name in possible_names:
                path = os.path.join(directory, name)
                if os.path.exists(path):
                    return path
                    
    return None

def main():
    model_path = "model/yolo11x.pt"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
        
    # 创建优化的检测器实例
    detector = OptimizedEllipseDetector(model_path, use_mps=True)
    
    # 询问用户选择处理模式
    print("请选择处理模式:")
    print("1. 处理视频文件")
    print("2. 使用摄像头实时处理")
    
    choice = input("请输入选项 (1 或 2): ").strip()
    
    if choice == "2":
        # 使用摄像头模式
        output_dir = "result"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "optimized_camera_output.mp4")
        
        print("使用摄像头模式...")
        print(f"模型: {model_path}")
        print(f"输出视频: {output_path}")
        
        try:
            detector.process_camera(output_path, confidence_threshold=0.3)
            print("完成!")
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 处理视频文件模式
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
            
        output_dir = "runs/optimized_detect"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "optimized_result.mp4")
        
        print("开始处理视频...")
        print(f"模型: {model_path}")
        print(f"输入视频: {video_path}")
        print(f"输出视频: {output_path}")
        
        try:
            detector.process_video(video_path, output_path, confidence_threshold=0.3)
            print("完成!")
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()