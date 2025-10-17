#!/usr/bin/env python3
"""
基于人体轮廓检测实现景深效果（背景虚化）
模拟单反相机M档拍照效果
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO  # type: ignore
import os
import argparse
from pathlib import Path

class DepthOfFieldEffect:
    def __init__(self, model_path="model/yolo11x-seg.pt", use_mps=True):
        """
        初始化景深效果处理器
        
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
        
    def create_depth_mask(self, frame, results, blur_strength=15):
        """
        创建景深效果的掩码
        
        Args:
            frame: 输入帧
            results: YOLO检测结果
            blur_strength: 背景模糊强度（高斯核大小）
            
        Returns:
            景深掩码
        """
        if not results or len(results) == 0:
            return frame
            
        result = results[0]
        if not hasattr(result, 'masks') or result.masks is None:
            return frame
            
        # 创建全尺寸的掩码
        depth_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        
        # 遍历所有检测到的对象
        for i, mask in enumerate(result.masks.data):
            # 只处理人体类别 (COCO数据集中类别0为person)
            class_id = int(result.boxes.cls[i])
            if class_id != 0:  # 0是'person'类
                continue
                
            # 将掩码转换为正确的形状
            mask = mask.cpu().numpy()
            # 调整掩码大小以匹配原始图像
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            # 累加到深度掩码中
            depth_mask = np.maximum(depth_mask, mask_resized)
        
        # 对掩码进行多级平滑处理，创建自然的过渡效果
        # 1. 使用双边滤波保持边缘同时平滑过渡区域
        depth_mask_uint8 = (depth_mask * 255).astype(np.uint8)
        depth_mask_bilateral = cv2.bilateralFilter(depth_mask_uint8, 15, 80, 80)
        
        # 2. 使用高斯模糊进一步软化边缘
        depth_mask_gaussian = cv2.GaussianBlur(depth_mask_bilateral, (15, 15), 0)
        depth_mask_smooth = depth_mask_gaussian.astype(np.float32) / 255.0
        
        # 3. 增强主体区域（使主体更清晰）
        # 使用更温和的gamma校正
        depth_mask_enhanced = np.power(depth_mask_smooth, 0.6)
        
        # 4. 应用距离变换来创建更平滑的边缘过渡
        binary_mask = (depth_mask_enhanced > 0.15).astype(np.uint8)
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        # 归一化距离变换
        dist_normalized = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # 5. 结合多种技术创建最终掩码
        # 主要使用平滑后的掩码，结合距离变换和原始掩码增强边缘
        depth_mask_final = 0.6 * depth_mask_enhanced + 0.3 * dist_normalized + 0.1 * depth_mask
        
        # 6. 应用S曲线调整使过渡更加自然
        depth_mask_scurve = np.power(depth_mask_final, 0.8)
        
        return depth_mask_scurve
        
    def apply_depth_of_field(self, frame, depth_mask, blur_strength=25):
        """
        应用景深效果到图像，最终优化版本避免亮线问题
        
        Args:
            frame: 输入帧
            depth_mask: 景深掩码
            blur_strength: 背景模糊强度
            
        Returns:
            应用景深效果后的图像
        """
        # 创建背景模糊版本
        # 根据blur_strength调整核大小（必须是奇数）
        kernel_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        kernel_size = max(3, kernel_size)  # 最小核大小为3
        
        # 对背景进行模糊处理
        blurred_background = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        # 优化掩码处理流程
        # 1. 确保掩码值在有效范围内
        depth_mask_clipped = np.clip(depth_mask, 0, 1)
        
        # 2. 应用中值滤波去除椒盐噪声
        mask_uint8 = (depth_mask_clipped * 255).astype(np.uint8)
        mask_denoised = cv2.medianBlur(mask_uint8, 3)
        
        # 3. 应用双边滤波保持边缘同时减少噪声
        mask_bilateral = cv2.bilateralFilter(mask_denoised, 9, 75, 75)
        
        # 4. 使用形态学操作平滑边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_morph = cv2.morphologyEx(mask_bilateral, cv2.MORPH_CLOSE, kernel)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, kernel)
        
        # 5. 转换回浮点数格式
        depth_mask_processed = mask_morph.astype(np.float32) / 255.0
        
        # 6. 应用Gamma校正优化过渡
        depth_mask_gamma = np.power(np.clip(depth_mask_processed, 0, 1), 0.75)
        
        # 7. 使用距离变换创建平滑边缘
        dist_transform = cv2.distanceTransform(mask_morph, cv2.DIST_L2, 5)
        dist_normalized = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # 8. 结合原始掩码和距离变换
        combined_mask = 0.7 * depth_mask_gamma + 0.3 * dist_normalized
        
        # 9. 应用额外平滑避免极端值
        combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 0)
        
        # 10. 确保数值范围正确
        final_mask = np.clip(combined_mask, 0, 1)
        
        # 11. 扩展为三通道
        mask_3channel = cv2.merge([final_mask, final_mask, final_mask])
        
        # 12. 应用S曲线调整使过渡更加自然
        transition_curve = np.power(np.clip(mask_3channel, 0, 1), 0.8)
        
        # 13. 确保数值在有效范围内
        transition_curve = np.clip(transition_curve, 0, 1)
        
        # 使用过渡曲线进行混合
        result = frame.astype(np.float32) * transition_curve + blurred_background.astype(np.float32) * (1 - transition_curve)
        
        # 14. 转换为uint8类型
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 15. 应用最终的亮线修复
        result = self._final_bright_line_fix(result)
        
        return result
    
    def _final_bright_line_fix(self, image):
        """
        最终亮线修复方法
        
        Args:
            image: 输入图像
            
        Returns:
            修复后的图像
        """
        # 创建结果图像
        result = image.copy().astype(np.float32)
        
        # 转换到LAB颜色空间分析亮度
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # 计算局部均值和标准差
        mean_l = cv2.blur(l_channel, (15, 15))
        squared_l = cv2.blur(l_channel.astype(np.float32) ** 2, (15, 15))
        std_l = np.sqrt(np.maximum(0, squared_l - mean_l.astype(np.float32) ** 2))
        
        # 检测异常亮的像素（超过局部均值+2倍标准差）
        bright_mask = l_channel > (mean_l + 2 * std_l)
        
        # 膨胀掩码以包含周围的像素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask_dilated = cv2.dilate(bright_mask.astype(np.uint8), kernel, iterations=1)
        
        # 对异常亮的区域应用平滑处理
        if np.any(bright_mask_dilated):
            # 创建平滑版本
            smoothed = cv2.bilateralFilter(image, 9, 75, 75)
            smoothed = cv2.GaussianBlur(smoothed, (3, 3), 0)
            
            # 只对异常亮的区域应用平滑
            bright_mask_3channel = cv2.merge([bright_mask_dilated, bright_mask_dilated, bright_mask_dilated])
            bright_mask_3channel = bright_mask_3channel.astype(np.float32) / 255.0
            
            # 混合原始图像和平滑图像
            result = result * (1 - bright_mask_3channel) + smoothed.astype(np.float32) * bright_mask_3channel
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    def process_image(self, image_path, output_path=None, blur_strength=25, confidence_threshold=0.5):
        """
        处理单张图片，应用景深效果
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径（如果为None，则显示结果）
            blur_strength: 背景模糊强度
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
        
        # 创建景深掩码
        print("创建景深掩码...")
        depth_mask = self.create_depth_mask(image, results, blur_strength)
        
        # 应用景深效果
        print("应用景深效果...")
        result_image = self.apply_depth_of_field(image, depth_mask, blur_strength)
        
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
            cv2.imshow('Depth of Field Effect', result_image)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return result_image
        
    def process_directory(self, input_dir, output_dir, blur_strength=25, confidence_threshold=0.5):
        """
        批量处理目录中的图片
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            blur_strength: 背景模糊强度
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
            output_file = os.path.join(output_dir, f"dof_{image_file.name}")
            
            try:
                self.process_image(str(image_file), output_file, blur_strength, confidence_threshold)
                print(f"  已保存到: {output_file}")
            except Exception as e:
                print(f"  处理失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='基于人体轮廓检测实现景深效果（背景虚化）')
    parser.add_argument('--input', '-i', type=str, required=True, 
                       help='输入图片路径或目录')
    parser.add_argument('--output', '-o', type=str, 
                       help='输出图片路径或目录')
    parser.add_argument('--model', '-m', type=str, default="model/yolo11x-seg.pt",
                       help='YOLO分割模型路径')
    parser.add_argument('--blur', '-b', type=int, default=25,
                       help='背景模糊强度 (默认: 25)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='禁用GPU加速')
    
    args = parser.parse_args()
    
    # 创建景深效果处理器（会自动检查并下载模型）
    use_mps = not args.no_gpu
    dof_processor = DepthOfFieldEffect(args.model, use_mps)
    
    # 检查输入是文件还是目录
    if os.path.isfile(args.input):
        # 处理单个文件
        dof_processor.process_image(args.input, args.output, args.blur, args.confidence)
    elif os.path.isdir(args.input):
        # 处理目录
        if not args.output:
            print("错误: 处理目录时必须指定输出目录")
            return
        dof_processor.process_directory(args.input, args.output, args.blur, args.confidence)
    else:
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    print("景深效果处理完成!")

if __name__ == "__main__":
    main()