#!/usr/bin/env python3
"""
景深效果改进展示
展示从基础版本到最终修复版本的所有改进
"""

import cv2
import numpy as np
import sys
import os

def create_improvement_showcase():
    """创建改进展示图"""
    print("=== 景深效果改进展示 ===")
    
    # 读取所有版本的结果
    original = cv2.imread("sample_bus.jpg")
    basic_result = cv2.imread("dof_test_result.jpg")
    improved_result = cv2.imread("dof_final_result.jpg")
    fixed_result = cv2.imread("dof_final_fixed_result.jpg")
    
    if original is None or basic_result is None or improved_result is None or fixed_result is None:
        print("错误: 无法读取必要的图像文件")
        return
    
    height, width = original.shape[:2]
    
    # 创建四联对比图
    comparison_width = width * 4
    comparison_height = height + 100  # 为标题留出空间
    
    comparison = np.zeros((comparison_height, comparison_width, 3), dtype=np.uint8)
    
    # 放置图像
    comparison[50:50+height, :width] = original
    comparison[50:50+height, width:2*width] = basic_result
    comparison[50:50+height, 2*width:3*width] = improved_result
    comparison[50:50+height, 3*width:4*width] = fixed_result
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "原始图像", (width//2-80, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "基础效果", (width + width//2-80, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "边缘优化", (2*width + width//2-80, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "亮线修复", (3*width + width//2-80, 30), font, 1, (0, 255, 0), 2)
    
    # 添加说明
    cv2.putText(comparison, "景深效果完整改进流程", (20, comparison_height-20), font, 0.7, (255, 255, 255), 1)
    
    cv2.imwrite("dof_complete_improvement_showcase.jpg", comparison)
    print("完整改进展示图已保存为 dof_complete_improvement_showcase.jpg")
    
    # 创建局部放大对比图以突出所有改进
    if width > 400 and height > 400:
        # 选择一个包含清晰边缘的区域
        crop_x = width - 350
        crop_y = 100
        crop_size = 300
        
        if crop_x + crop_size <= width and crop_y + crop_size <= height:
            # 提取局部区域
            regions = [
                original[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size],
                basic_result[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size],
                improved_result[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size],
                fixed_result[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
            ]
            
            # 放大2倍
            zoom_factor = 2
            zoomed_regions = []
            for region in regions:
                zoomed = cv2.resize(region, 
                                  (crop_size * zoom_factor, crop_size * zoom_factor), 
                                  interpolation=cv2.INTER_LINEAR)
                zoomed_regions.append(zoomed)
            
            # 创建放大对比图
            zoom_width = crop_size * zoom_factor * 4
            zoom_height = crop_size * zoom_factor + 80
            
            zoom_comparison = np.zeros((zoom_height, zoom_width, 3), dtype=np.uint8)
            for i, zoomed_region in enumerate(zoomed_regions):
                zoom_comparison[50:50+crop_size * zoom_factor, 
                              i * crop_size * zoom_factor:(i + 1) * crop_size * zoom_factor] = zoomed_region
            
            # 添加标签
            labels = ["原始", "基础", "优化", "修复"]
            for i, label in enumerate(labels):
                x_pos = i * crop_size * zoom_factor + crop_size * zoom_factor//2 - 50
                cv2.putText(zoom_comparison, label, (x_pos, 30), font, 0.8, (0, 255, 0), 2)
            
            cv2.imwrite("dof_complete_improvement_zoom.jpg", zoom_comparison)
            print("完整改进放大对比图已保存为 dof_complete_improvement_zoom.jpg")

def create_final_comparison_report():
    """创建最终对比报告"""
    print("\n=== 创建最终对比报告 ===")
    
    # 读取图像
    original = cv2.imread("sample_bus.jpg")
    basic_result = cv2.imread("dof_test_result.jpg")
    fixed_result = cv2.imread("dof_final_fixed_result.jpg")
    
    if original is None or basic_result is None or fixed_result is None:
        print("错误: 无法读取图像文件")
        return
    
    # 计算图像质量指标
    # 1. PSNR (峰值信噪比)
    def calculate_psnr(img1, img2):
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    # 2. SSIM (结构相似性指数) - 简化版本
    def calculate_ssim(img1, img2):
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 计算均值
        mean1 = np.mean(gray1)
        mean2 = np.mean(gray2)
        
        # 计算方差
        var1 = np.var(gray1)
        var2 = np.var(gray2)
        
        # 计算协方差
        covar = np.mean((gray1 - mean1) * (gray2 - mean2))
        
        # SSIM参数
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
               ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))
        
        return ssim
    
    # 计算指标
    psnr_basic = calculate_psnr(original, basic_result)
    psnr_fixed = calculate_psnr(original, fixed_result)
    
    ssim_basic = calculate_ssim(original, basic_result)
    ssim_fixed = calculate_ssim(original, fixed_result)
    
    print("图像质量对比:")
    print(f"基础版本 PSNR: {psnr_basic:.2f} dB")
    print(f"修复版本 PSNR: {psnr_fixed:.2f} dB")
    print(f"PSNR 提升: {psnr_fixed - psnr_basic:.2f} dB")
    
    print(f"基础版本 SSIM: {ssim_basic:.4f}")
    print(f"修复版本 SSIM: {ssim_fixed:.4f}")
    print(f"SSIM 提升: {ssim_fixed - ssim_basic:.4f}")
    
    # 创建报告图像
    height, width = original.shape[:2]
    report = np.zeros((200, 800, 3), dtype=np.uint8)
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(report, "景深效果改进报告", (20, 40), font, 1, (0, 255, 0), 2)
    
    # 添加指标
    y_pos = 80
    metrics = [
        f"基础版本 PSNR: {psnr_basic:.2f} dB",
        f"修复版本 PSNR: {psnr_fixed:.2f} dB",
        f"PSNR 提升: {psnr_fixed - psnr_basic:.2f} dB",
        f"基础版本 SSIM: {ssim_basic:.4f}",
        f"修复版本 SSIM: {ssim_fixed:.4f}",
        f"SSIM 提升: {ssim_fixed - ssim_basic:.4f}"
    ]
    
    for metric in metrics:
        cv2.putText(report, metric, (20, y_pos), font, 0.6, (255, 255, 255), 1)
        y_pos += 25
    
    cv2.imwrite("dof_improvement_report.jpg", report)
    print("改进报告已保存为 dof_improvement_report.jpg")

if __name__ == "__main__":
    create_improvement_showcase()
    create_final_comparison_report()
    print("\n景深效果改进展示完成!")