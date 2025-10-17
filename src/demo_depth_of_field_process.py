#!/usr/bin/env python3
"""
演示景深效果处理过程
"""

import cv2
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

def demonstrate_dof_process():
    """演示景深效果处理过程"""
    print("=== 景深效果处理过程演示 ===")
    
    # 读取测试结果
    original_image = cv2.imread("sample_bus.jpg")
    dof_result = cv2.imread("dof_test_result.jpg")
    
    if original_image is None or dof_result is None:
        print("错误: 无法读取测试图像")
        print("请先运行测试脚本生成结果图像")
        return
    
    # 显示原始图像信息
    print(f"原始图像尺寸: {original_image.shape[1]}x{original_image.shape[0]}")
    print(f"景深效果图像尺寸: {dof_result.shape[1]}x{dof_result.shape[0]}")
    
    # 创建处理过程演示图
    height, width = original_image.shape[:2]
    
    # 创建演示图像（包含原始图像、掩码示意、景深效果）
    demo_width = width * 3
    demo_height = height + 100  # 额外空间用于标题
    
    demo_image = np.zeros((demo_height, demo_width, 3), dtype=np.uint8)
    
    # 复制原始图像
    demo_image[50:50+height, :width] = original_image
    
    # 中间位置放置掩码示意（模拟）
    mask_demo = original_image.copy()
    # 在图像中央绘制一个半透明的矩形来示意主体区域
    overlay = mask_demo.copy()
    cv2.rectangle(overlay, (width//4, height//4), (3*width//4, 3*height//4), (0, 255, 0), -1)
    mask_demo = cv2.addWeighted(overlay, 0.3, mask_demo, 0.7, 0)
    demo_image[50:50+height, width:2*width] = mask_demo
    
    # 复制景深效果图像
    demo_image[50:50+height, 2*width:3*width] = dof_result
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(demo_image, "原始图像", (width//2-80, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(demo_image, "主体掩码", (width + width//2-80, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(demo_image, "景深效果", (2*width + width//2-80, 30), font, 1, (0, 255, 0), 2)
    
    # 添加说明文字
    cv2.putText(demo_image, "基于YOLOv8-seg实例分割实现", (20, demo_height-20), font, 0.6, (255, 255, 255), 1)
    
    # 保存演示图像
    cv2.imwrite("dof_process_demo.jpg", demo_image)
    print("景深效果处理过程演示图已保存为 dof_process_demo.jpg")
    
    # 创建不同模糊强度的对比图
    blur_images = []
    blur_strengths = [15, 25, 40]
    
    for strength in blur_strengths:
        image_path = f"dof_blur_{strength}.jpg"
        img = cv2.imread(image_path)
        if img is not None:
            # 添加强度标签
            cv2.putText(img, f"模糊强度: {strength}", (20, 40), font, 1, (0, 255, 0), 2)
            blur_images.append(img)
    
    if len(blur_images) >= 3:
        # 创建对比图像
        comparison_height = max(img.shape[0] for img in blur_images)
        comparison_width = sum(img.shape[1] for img in blur_images)
        
        comparison_image = np.zeros((comparison_height, comparison_width, 3), dtype=np.uint8)
        
        # 水平拼接图像
        x_offset = 0
        for img in blur_images:
            h, w = img.shape[:2]
            comparison_image[:h, x_offset:x_offset+w] = img
            x_offset += w
        
        # 保存对比图像
        cv2.imwrite("dof_blur_strength_comparison.jpg", comparison_image)
        print("不同模糊强度对比图已保存为 dof_blur_strength_comparison.jpg")
    
    print("\n演示完成!")
    print("生成的文件:")
    print("  1. dof_process_demo.jpg - 景深效果处理过程演示图")
    print("  2. dof_blur_strength_comparison.jpg - 不同模糊强度对比图")

if __name__ == "__main__":
    demonstrate_dof_process()