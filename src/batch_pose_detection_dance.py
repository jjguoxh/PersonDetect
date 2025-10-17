#!/usr/bin/env python3
"""
批量处理dance目录下的所有舞蹈视频，检测人体关键点
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
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# 关键点连接关系（用于可视化骨骼）
SKELETON_CONNECTIONS = [
    # 头部
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 身体
    (5, 6), (5, 11), (6, 12), (11, 12),
    # 手臂
    (5, 7), (7, 9), (6, 8), (8, 10),
    # 腿部
    (11, 13), (13, 15), (12, 14), (14, 16)
]

class BatchDancePoseDetector:
    def __init__(self, model_path="yolo11n-pose.pt", use_mps=False):
        """
        初始化批量姿态检测器
        
        Args:
            model_path: 姿态估计模型路径
            use_mps: 是否使用MPS加速（注意：姿态估计在MPS上可能有问题）
        """
        # 注意：在姿态估计模式下，MPS可能存在已知问题，推荐使用CPU
        self.device = 'cpu'  # 强制使用CPU以避免MPS相关问题
        print(f"使用设备: {self.device}")
        
        # 加载姿态估计模型
        print(f"加载姿态估计模型: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def draw_keypoints(self, frame, keypoints, confidences, threshold=0.5):
        """
        在帧上绘制关键点和骨骼连接
        
        Args:
            frame: 视频帧
            keypoints: 关键点坐标 [17, 2]
            confidences: 关键点置信度 [17]
            threshold: 关键点置信度阈值
        """
        h, w = frame.shape[:2]
        
        # 绘制骨骼连接
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if (confidences[start_idx] > threshold and 
                confidences[end_idx] > threshold):
                start_point = (int(keypoints[start_idx][0] * w), 
                              int(keypoints[start_idx][1] * h))
                end_point = (int(keypoints[end_idx][0] * w), 
                            int(keypoints[end_idx][1] * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # 绘制关键点
        for i, (x, y) in enumerate(keypoints):
            if confidences[i] > threshold:
                px, py = int(x * w), int(y * h)
                # 根据关键点类型使用不同颜色
                if i in [0, 1, 2, 3, 4]:  # 头部关键点
                    color = (0, 0, 255)  # 红色
                elif i in [5, 6, 7, 8, 9, 10]:  # 上肢关键点
                    color = (255, 0, 0)  # 蓝色
                else:  # 下肢关键点
                    color = (0, 255, 255)  # 黄色
                    
                cv2.circle(frame, (px, py), 5, color, -1)
                cv2.circle(frame, (px, py), 6, (255, 255, 255), 2)
                
                # 添加关键点标签
                keypoint_name = list(KEYPOINT_NAMES.keys())[list(KEYPOINT_NAMES.values()).index(i)]
                cv2.putText(frame, keypoint_name, (px + 10, py + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def process_frame(self, frame, confidence_threshold=0.5):
        """
        处理单个视频帧
        
        Args:
            frame: 输入视频帧
            confidence_threshold: 检测置信度阈值
            
        Returns:
            processed_frame: 处理后的帧
            person_count: 检测到的人数
            keypoints_data: 关键点数据列表
        """
        # 使用姿态估计模型进行检测
        with torch.no_grad():
            results = self.model(frame, verbose=False, device=self.device, 
                               conf=confidence_threshold)
        
        processed_frame = frame.copy()
        person_count = 0
        keypoints_data = []
        
        # 处理检测结果
        if results and len(results) > 0:
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    # 获取关键点数据
                    keypoints_data_batch = result.keypoints.data
                    
                    # 处理每个检测到的人
                    for person_keypoints in keypoints_data_batch:
                        person_count += 1
                        # 提取关键点坐标和置信度
                        keypoints_xy = person_keypoints[:, :2].cpu().numpy()  # [17, 2]
                        confidences = person_keypoints[:, 2].cpu().numpy()    # [17]
                        
                        # 保存关键点数据
                        keypoints_data.append({
                            'keypoints': keypoints_xy,
                            'confidences': confidences
                        })
                        
                        # 绘制关键点和连接
                        self.draw_keypoints(processed_frame, keypoints_xy, confidences, 
                                          threshold=0.3)
                        
        return processed_frame, person_count, keypoints_data
    
    def process_video(self, video_path, output_path, confidence_threshold=0.5):
        """
        处理整个视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            confidence_threshold: 置信度阈值
            
        Returns:
            dict: 处理统计信息
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
        total_persons = 0
        start_time = time.time()
        
        print("开始处理视频帧...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 处理帧
            processed_frame, person_count, keypoints_data = self.process_frame(frame, confidence_threshold)
            
            # 累计检测到的人数
            total_persons += person_count
            
            # 添加帧信息
            cv2.putText(processed_frame, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f'Persons: {person_count}', (10, 70), 
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
        avg_persons_per_frame = total_persons / frame_count if frame_count > 0 else 0
        
        stats = {
            'total_frames': frame_count,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'total_persons': total_persons,
            'avg_persons_per_frame': avg_persons_per_frame,
            'output_path': output_path,
            'output_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
        }
        
        print(f"视频处理完成!")
        print(f"总帧数: {frame_count}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均处理速度: {avg_fps:.2f} FPS")
        print(f"检测到的总人数: {total_persons}")
        print(f"平均每帧人数: {avg_persons_per_frame:.2f}")
        print(f"输出文件: {output_path}")
        print(f"输出文件大小: {stats['output_size']} 字节")
        
        return stats
    
    def process_all_videos(self, dance_dir="dance", output_base_dir="runs/batch_dance_pose", 
                          confidence_threshold=0.5):
        """
        批量处理dance目录下的所有视频文件
        
        Args:
            dance_dir: 舞蹈视频目录
            output_base_dir: 输出基础目录
            confidence_threshold: 置信度阈值
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
        results = []
        for i, video_path in enumerate(video_files):
            print(f"\n处理第 {i+1}/{len(video_files)} 个视频...")
            video_name = Path(video_path).stem
            output_path = os.path.join(output_base_dir, f"{video_name}_pose_result.mp4")
            
            try:
                stats = self.process_video(video_path, output_path, confidence_threshold)
                stats['input_video'] = video_path
                results.append(stats)
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
                
        # 输出处理总结
        print("\n" + "="*50)
        print("批量处理完成总结:")
        print("="*50)
        total_frames = sum(r['total_frames'] for r in results)
        total_time = sum(r['total_time'] for r in results)
        total_persons = sum(r['total_persons'] for r in results)
        
        print(f"处理视频数量: {len(results)}")
        print(f"总帧数: {total_frames}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"检测到的总人数: {total_persons}")
        print(f"平均每帧人数: {total_persons/total_frames if total_frames > 0 else 0:.2f}")
        
        print("\n各视频处理详情:")
        for result in results:
            print(f"- {os.path.basename(result['input_video'])}: "
                  f"{result['total_frames']} 帧, "
                  f"{result['total_persons']} 人, "
                  f"{result['avg_fps']:.2f} FPS")

def main():
    # 检查是否有预训练的姿态估计模型
    pose_model_path = "model/yolo11n-pose.pt"
    
    # 如果没有找到姿态模型，尝试下载
    if not os.path.exists(pose_model_path):
        print(f"未找到姿态估计模型 {pose_model_path}，将自动下载...")
        # Ultralytics会自动下载模型
    
    # 创建批量姿态检测器实例
    detector = BatchDancePoseDetector(pose_model_path, use_mps=False)
    
    print("开始批量处理dance目录下的所有视频...")
    print(f"姿态模型: {pose_model_path}")
    
    try:
        detector.process_all_videos(confidence_threshold=0.3)
        print("\n所有视频处理完成!")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()