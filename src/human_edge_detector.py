#!/usr/bin/env python3
"""
使用YOLOv8实例分割检测人体边缘并标记
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO  # type: ignore
import os
import argparse
from pathlib import Path

class HumanEdgeDetector:
    def __init__(self, model_path="model/yolo11x-seg.pt", use_mps=True):
        """
        初始化人体边缘检测器
        
        Args:
            model_path: YOLO分割模型路径
            use_mps: 是否使用MPS加速
        """
        # 检查MPS可用性（适用于Apple Silicon）
        self.device = 'mps' if use_mps and torch.backends.mps.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 检查并下载模型
        self.model_path = self._check_and_download_model(model_path)
        
        # 加载分割模型
        print(f"加载分割模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def _check_and_download_model(self, model_path):
        """
        检查模型文件是否存在，不存在则自动下载
        
        Args:
            model_path: 模型路径
            
        Returns:
            模型路径
        """
        # 如果模型文件已存在，直接返回路径
        if os.path.exists(model_path):
            print(f"模型文件已存在: {model_path}")
            return model_path
            
        # 确保模型目录存在
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            
        # 获取模型名称
        model_name = os.path.basename(model_path)
        print(f"模型文件不存在，正在下载: {model_name}")
        
        try:
            # 使用Ultralytics自动下载模型
            model = YOLO(model_name)
            print(f"模型下载完成!")
            return model_path
        except Exception as e:
            print(f"模型下载失败: {e}")
            # 如果指定路径的模型下载失败，尝试使用默认模型
            default_model = "yolo11x-seg.pt"
            print(f"尝试下载默认模型: {default_model}")
            try:
                model = YOLO(default_model)
                # 移动到指定路径
                if os.path.exists(default_model):
                    import shutil
                    shutil.move(default_model, model_path)
                print(f"默认模型下载完成并移动到: {model_path}")
                return model_path
            except Exception as e2:
                raise RuntimeError(f"无法下载模型: {e2}")
        
    def draw_human_masks(self, frame, results, alpha=0.5):
        """
        在帧上绘制人体掩码，特别优化头发等细节边缘
        
        Args:
            frame: 输入帧
            results: YOLO检测结果
            alpha: 掩码透明度
            
        Returns:
            带有掩码标记的帧
        """
        if not results or len(results) == 0:
            return frame
            
        result = results[0]
        if not hasattr(result, 'masks') or result.masks is None:
            return frame
            
        # 创建结果帧
        result_frame = frame.copy()
        
        # 遍历所有检测到的对象
        for i, mask in enumerate(result.masks.data):
            # 只处理人体类别 (COCO数据集中类别0为person)
            class_id = int(result.boxes.cls[i])
            if class_id != 0:  # 0是'person'类
                continue
                
            # 将掩码转换为正确的形状
            mask = mask.cpu().numpy()
            # 调整掩码大小以匹配原始图像
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            # 高级边缘处理，特别针对头发细节
            # 1. 使用双边滤波保持边缘同时减少噪声
            mask_filtered = cv2.bilateralFilter((mask * 255).astype(np.uint8), 9, 75, 75)
            mask_filtered = mask_filtered.astype(np.float32) / 255.0
            
            # 2. 使用自适应阈值分割获得更好的边缘
            mask_uint8 = (mask_filtered * 255).astype(np.uint8)
            mask_adaptive = cv2.adaptiveThreshold(mask_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            mask_adaptive = mask_adaptive.astype(np.float32) / 255.0
            
            # 3. 结合原始掩码和自适应掩码
            mask_combined = 0.7 * mask_filtered + 0.3 * mask_adaptive
            mask_combined = np.clip(mask_combined, 0, 1)
            
            # 4. 使用距离变换增强边缘连续性
            mask_binary = (mask_combined > 0.5).astype(np.uint8)
            dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
            dist_normalized = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
            
            # 5. 创建多级边缘掩码
            mask_levels = {}
            thresholds = [0.3, 0.5, 0.7, 0.85]
            for j, thresh in enumerate(thresholds):
                mask_levels[j] = (mask_combined > thresh).astype(np.uint8)
            
            # 6. 特殊处理头发细节
            # 使用细化的边缘检测
            edges = cv2.Canny((mask_combined * 255).astype(np.uint8), 10, 50)
            # 使用形态学操作连接断开的边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 创建彩色掩码
            base_color = np.random.randint(0, 255, (3,)).tolist()
            
            # 使用渐变色填充区域
            overlay = result_frame.copy()
            colored_mask = np.full_like(overlay, base_color, dtype=np.uint8)
            
            # 使用距离变换作为透明度权重
            dist_3channel = cv2.merge([dist_normalized, dist_normalized, dist_normalized])
            mask_region = np.where(dist_3channel > 0.1, 
                                  cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0), 
                                  overlay)
            
            # 绘制多级轮廓
            contour_colors = [
                base_color,  # 外层轮廓
                [min(255, c + 20) for c in base_color],  # 中层轮廓
                [min(255, c + 40) for c in base_color],  # 内层轮廓
                [min(255, c + 60) for c in base_color]   # 核心轮廓
            ]
            
            # 绘制不同级别的轮廓
            for level, thresh in enumerate(thresholds):
                contours, _ = cv2.findContours(mask_levels[level], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                # 根据级别调整线条粗细
                thickness = max(1, 3 - level)
                cv2.drawContours(mask_region, contours, -1, contour_colors[level], thickness, cv2.LINE_AA)
            
            # 特别处理头发细节轮廓
            hair_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hair_color = [min(255, c + 50) for c in base_color]  # 更亮的颜色突出头发细节
            cv2.drawContours(mask_region, hair_contours, -1, hair_color, 1, cv2.LINE_AA)
            
            # 更新结果帧
            result_frame = mask_region
        
        # 将最终结果与原始帧混合
        cv2.addWeighted(result_frame, alpha, frame, 1 - alpha, 0, frame)
        return frame
        
    def process_image(self, image_path, output_path=None, confidence_threshold=0.5):
        """
        处理单张图片
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径（如果为None，则显示结果）
            confidence_threshold: 置信度阈值
        """
        print(f"加载图片: {image_path}")
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
            
        print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 使用模型进行分割
        print("执行人体分割...")
        with torch.no_grad():
            results = self.model(image, verbose=False, device=self.device, conf=confidence_threshold)
        
        # 绘制掩码
        result_image = self.draw_human_masks(image.copy(), results)
        
        # 保存或显示结果
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            cv2.imwrite(output_path, result_image)
            print(f"结果已保存到: {output_path}")
        else:
            # 显示结果
            cv2.imshow('Human Edge Detection', result_image)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return result_image
        
    def process_directory(self, input_dir, output_dir, confidence_threshold=0.5):
        """
        批量处理目录中的图片
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            confidence_threshold: 置信度阈值
        """
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # 获取所有图片文件
        input_path = Path(input_dir)
        image_files = [f for f in input_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"在目录 {input_dir} 中未找到图片文件")
            return
            
        print(f"找到 {len(image_files)} 个图片文件")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理每个图片
        for i, image_file in enumerate(image_files):
            print(f"处理图片 ({i+1}/{len(image_files)}): {image_file.name}")
            
            # 生成输出路径
            output_file = os.path.join(output_dir, f"edge_{image_file.name}")
            
            try:
                self.process_image(str(image_file), output_file, confidence_threshold)
                print(f"  已保存到: {output_file}")
            except Exception as e:
                print(f"  处理失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='使用YOLO检测人体边缘')
    parser.add_argument('--input', '-i', type=str, required=True, 
                       help='输入图片路径或目录')
    parser.add_argument('--output', '-o', type=str, 
                       help='输出图片路径或目录')
    parser.add_argument('--model', '-m', type=str, default="model/yolo11x-seg.pt",
                       help='YOLO分割模型路径')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='禁用GPU加速')
    
    args = parser.parse_args()
    
    # 创建检测器（会自动检查并下载模型）
    use_mps = not args.no_gpu
    detector = HumanEdgeDetector(args.model, use_mps)
    
    # 检查输入是文件还是目录
    if os.path.isfile(args.input):
        # 处理单个文件
        detector.process_image(args.input, args.output, args.confidence)
    elif os.path.isdir(args.input):
        # 处理目录
        if not args.output:
            print("错误: 处理目录时必须指定输出目录")
            return
        detector.process_directory(args.input, args.output, args.confidence)
    else:
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    print("处理完成!")

if __name__ == "__main__":
    main()