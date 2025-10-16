#!/usr/bin/env python3
"""
最终版舞蹈视频姿态检测器
检测视频中人体的头部、肩膀、肘部、腕部等大关节位置
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
from pathlib import Path

# 人体关键点索引映射
KEYPOINT_NAMES = {
    0: 'nose',          # 鼻子
    1: 'left_eye',      # 左眼
    2: 'right_eye',     # 右眼
    3: 'left_ear',      # 左耳
    4: 'right_ear',     # 右耳
    5: 'left_shoulder', # 左肩
    6: 'right_shoulder',# 右肩
    7: 'left_elbow',    # 左肘
    8: 'right_elbow',   # 右肘
    9: 'left_wrist',    # 左腕
    10: 'right_wrist',  # 右腕
    11: 'left_hip',     # 左髋
    12: 'right_hip',    # 右髋
    13: 'left_knee',    # 左膝
    14: 'right_knee',   # 右膝
    15: 'left_ankle',   # 左踝
    16: 'right_ankle'   # 右踝
}

# 关键点连接关系（用于可视化骨骼）
SKELETON_CONNECTIONS = [
    # 头部连接
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 身体连接
    (5, 6), (5, 11), (6, 12), (11, 12),
    # 手臂连接
    (5, 7), (7, 9), (6, 8), (8, 10),
    # 腿部连接
    (11, 13), (13, 15), (12, 14), (14, 16)
]

class DancePoseDetector:
    def __init__(self, model_path="yolo11n-pose.pt"):
        """
        初始化舞蹈姿态检测器
        
        Args:
            model_path: 姿态估计模型路径
        """
        # 强制使用CPU以避免MPS相关问题
        self.device = 'cpu'
        print(f"使用设备: {self.device}")
        
        # 加载姿态估计模型
        print(f"加载姿态估计模型: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def draw_skeleton(self, frame, keypoints, confidences, threshold=0.5):
        """
        在帧上绘制骨骼连接和关键点
        
        Args:
            frame: 视频帧
            keypoints: 关键点坐标 [17, 2] (像素坐标)
            confidences: 关键点置信度 [17]
            threshold: 关键点置信度阈值
        """
        h, w = frame.shape[:2]
        
        # 绘制骨骼连接线
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            # 确保索引有效且置信度足够高
            if (0 <= start_idx < len(confidences) and 
                0 <= end_idx < len(confidences) and
                confidences[start_idx] > threshold and 
                confidences[end_idx] > threshold):
                start_point = (int(keypoints[start_idx][0]), 
                              int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), 
                            int(keypoints[end_idx][1]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 3)
        
        # 绘制关键点
        for i, (x, y) in enumerate(keypoints):
            # 确保索引有效且置信度足够高
            if 0 <= i < len(confidences) and confidences[i] > threshold:
                px, py = int(x), int(y)
                
                # 根据关键点类型使用不同颜色
                if i in [0, 1, 2, 3, 4]:  # 头部关键点 (红色)
                    color = (0, 0, 255)
                elif i in [5, 6, 7, 8, 9, 10]:  # 上肢关键点 (蓝色)
                    color = (255, 0, 0)
                else:  # 下肢关键点 (黄色)
                    color = (0, 255, 255)
                
                # 绘制关键点圆圈
                cv2.circle(frame, (px, py), 6, color, -1)
                cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)
                
                # 添加关键点标签
                if i in KEYPOINT_NAMES:
                    label = KEYPOINT_NAMES[i]
                    cv2.putText(frame, label, (px + 10, py - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def process_frame(self, frame, confidence_threshold=0.5):
        """
        处理单个视频帧
        
        Args:
            frame: 输入视频帧
            confidence_threshold: 检测置信度阈值
            
        Returns:
            processed_frame: 处理后的帧
            person_count: 检测到的人数
        """
        # 使用姿态估计模型进行检测
        with torch.no_grad():
            results = self.model(frame, verbose=False, device=self.device, 
                               conf=confidence_threshold)
        
        processed_frame = frame.copy()
        person_count = 0
        
        # 处理检测结果
        if results and len(results) > 0:
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints
                    
                    # 处理每个检测到的人
                    for i in range(len(keypoints_data)):
                        person_count += 1
                        # 提取关键点坐标和置信度
                        # 注意：这里直接使用像素坐标，而不是归一化坐标
                        keypoints_xy = keypoints_data[i].xy.cpu().numpy()[0]  # [17, 2] 像素坐标
                        confidences = keypoints_data[i].conf.cpu().numpy()[0] if keypoints_data[i].conf is not None else np.ones(17)  # [17]
                        
                        # 绘制骨骼和关键点
                        self.draw_skeleton(processed_frame, keypoints_xy, confidences, 
                                         threshold=0.3)
                        
        return processed_frame, person_count
    
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
            processed_frame, person_count = self.process_frame(frame, confidence_threshold)
            
            # 添加帧信息
            cv2.putText(processed_frame, f'Frame: {frame_count}', (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f'Persons: {person_count}', (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 写入处理后的帧
            out.write(processed_frame)
            frame_count += 1
            
            # 显示进度
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps_processed = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"已处理 {frame_count}/{total_frames} 帧, "
                      f"当前帧检测到 {person_count} 个人, "
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
    
    def process_all_dance_videos(self, dance_dir="dance", 
                                output_base_dir="runs/final_dance_pose"):
        """
        批量处理dance目录下的所有舞蹈视频
        
        Args:
            dance_dir: 舞蹈视频目录
            output_base_dir: 输出基础目录
        """
        if not os.path.exists(dance_dir):
            raise ValueError(f"找不到目录: {dance_dir}")
            
        # 获取所有视频文件
        video_files = []
        for file in os.listdir(dance_dir):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(dance_dir, file))
                
        if not video_files:
            print(f"在目录 {dance_dir} 中未找到视频文件")
            return
            
        print(f"找到 {len(video_files)} 个视频文件:")
        for i, video_file in enumerate(video_files):
            print(f"{i+1}. {video_file}")
            
        # 创建输出目录
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 处理每个视频文件
        for i, video_path in enumerate(video_files):
            print(f"\n处理第 {i+1}/{len(video_files)} 个视频: {os.path.basename(video_path)}")
            video_name = Path(video_path).stem
            output_path = os.path.join(output_base_dir, f"{video_name}_pose_result.mp4")
            
            try:
                self.process_video(video_path, output_path, confidence_threshold=0.3)
                print(f"完成处理: {video_path}")
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                import traceback
                traceback.print_exc()

def main():
    # 检查模型文件
    pose_model_path = "yolo11n-pose.pt"
    
    # 创建姿态检测器实例
    print("初始化舞蹈姿态检测器...")
    detector = DancePoseDetector(pose_model_path)
    
    # 询问用户选择处理模式
    print("\n请选择处理模式:")
    print("1. 处理dance目录下的单个视频文件")
    print("2. 批量处理dance目录下的所有视频")
    print("3. 使用摄像头实时处理")
    
    try:
        choice = input("请输入选项 (1-3): ").strip()
    except KeyboardInterrupt:
        print("\n用户取消操作")
        return
    
    if choice == "3":
        # 使用摄像头模式
        output_dir = "result"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "dance_pose_camera_output.mp4")
        
        print("使用摄像头模式...")
        print(f"姿态模型: {pose_model_path}")
        print(f"输出视频: {output_path}")
        
        try:
            # 注意：摄像头模式需要实时处理，这里简化处理
            print("摄像头模式需要实时处理，建议使用单独的实时处理脚本")
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    elif choice == "2":
        # 批量处理所有视频
        print("开始批量处理dance目录下的所有视频...")
        print(f"姿态模型: {pose_model_path}")
        
        try:
            detector.process_all_dance_videos()
            print("\n所有视频处理完成!")
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 处理单个视频文件
        dance_dir = "dance"
        if not os.path.exists(dance_dir):
            print(f"错误: 找不到目录 {dance_dir}")
            return
            
        video_files = []
        for file in os.listdir(dance_dir):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(dance_dir, file))
                
        if not video_files:
            print("错误: 在dance目录下找不到视频文件")
            return
            
        print("找到以下视频文件:")
        for i, video_file in enumerate(video_files):
            print(f"{i+1}. {video_file}")
            
        try:
            selection = int(input(f"请选择要处理的视频文件 (1-{len(video_files)}): ")) - 1
            if selection < 0 or selection >= len(video_files):
                print("无效的选择")
                return
            video_path = video_files[selection]
        except ValueError:
            print("无效的输入")
            return
        except KeyboardInterrupt:
            print("\n用户取消操作")
            return
            
        # 创建输出目录
        output_dir = "runs/final_dance_pose"
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_name}_pose_result.mp4")
        
        print("开始处理视频...")
        print(f"姿态模型: {pose_model_path}")
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