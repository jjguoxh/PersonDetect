#!/usr/bin/env python3
"""
分析和修复景深效果中的亮线问题
"""

import cv2
import numpy as np
import sys
import os

def detect_bright_lines(image1, image2):
    """检测两张图像之间的亮线差异"""
    # 转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # 计算差异
    diff = cv2.absdiff(gray1, gray2)
    
    # 应用阈值检测显著差异区域
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # 形态学操作连接相邻区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def analyze_bright_line_issues():
    """分析亮线问题"""
    print("=== 分析亮线问题 ===")
    
    # 读取图像
    original = cv2.imread("sample_bus.jpg")
    dof_result = cv2.imread("dof_final_result.jpg")
    
    if original is None or dof_result is None:
        print("错误: 无法读取图像文件")
        return
    
    # 检测亮线区域
    bright_lines = detect_bright_lines(original, dof_result)
    
    # 在原始图像上标记亮线区域
    original_marked = original.copy()
    bright_lines_3channel = cv2.merge([bright_lines, bright_lines, bright_lines])
    original_marked = np.where(bright_lines_3channel > 0, 
                              [0, 0, 255],  # 红色标记
                              original_marked)
    
    # 保存分析结果
    cv2.imwrite("bright_lines_detected.jpg", bright_lines)
    cv2.imwrite("original_with_bright_lines_marked.jpg", original_marked)
    
    # 统计亮线区域
    bright_pixels = np.sum(bright_lines > 0)
    total_pixels = bright_lines.shape[0] * bright_lines.shape[1]
    bright_ratio = bright_pixels / total_pixels * 100
    
    print(f"检测到的亮线像素数: {bright_pixels}")
    print(f"亮线像素占比: {bright_ratio:.4f}%")
    
    # 创建详细分析图
    height, width = original.shape[:2]
    analysis = np.zeros((height, width * 3, 3), dtype=np.uint8)
    
    analysis[:, :width] = original
    analysis[:, width:2*width] = dof_result
    analysis[:, 2*width:] = cv2.merge([bright_lines, bright_lines, bright_lines])
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(analysis, "原始图像", (50, 50), font, 1, (0, 255, 0), 2)
    cv2.putText(analysis, "景深效果", (width + 50, 50), font, 1, (0, 255, 0), 2)
    cv2.putText(analysis, "亮线区域", (2*width + 50, 50), font, 1, (0, 255, 0), 2)
    
    cv2.imwrite("bright_line_analysis_detailed.jpg", analysis)
    print("详细分析图已保存为 bright_line_analysis_detailed.jpg")

def fix_bright_lines_simple(image, mask_threshold=200):
    """简单的亮线修复方法"""
    # 转换到LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # 检测过亮区域
    bright_mask = l_channel > mask_threshold
    
    # 创建结果图像
    result = image.copy()
    
    # 对过亮区域应用局部均值
    if np.any(bright_mask):
        # 膨胀掩码以包含周围区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask_dilated = cv2.dilate(bright_mask.astype(np.uint8), kernel, iterations=1)
        
        # 对每个通道应用平滑
        for i in range(3):
            # 应用双边滤波
            smoothed = cv2.bilateralFilter(result[:, :, i], 9, 75, 75)
            # 只更新过亮区域
            result[:, :, i] = np.where(bright_mask_dilated > 0, 
                                     smoothed, 
                                     result[:, :, i])
    
    return result

def test_simple_fix():
    """测试简单的亮线修复方法"""
    print("\n=== 测试简单的亮线修复方法 ===")
    
    # 读取景深效果结果
    dof_result = cv2.imread("dof_final_result.jpg")
    if dof_result is None:
        print("错误: 无法读取景深效果结果")
        return
    
    # 应用修复
    fixed_result = fix_bright_lines_simple(dof_result, mask_threshold=200)
    
    # 保存修复结果
    cv2.imwrite("dof_simple_fix_result.jpg", fixed_result)
    print("简单修复结果已保存为 dof_simple_fix_result.jpg")
    
    # 创建对比图
    height, width = dof_result.shape[:2]
    comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
    comparison[:, :width] = dof_result
    comparison[:, width:] = fixed_result
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "修复前", (50, 50), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "修复后", (width + 50, 50), font, 1, (0, 255, 0), 2)
    
    cv2.imwrite("dof_simple_fix_comparison.jpg", comparison)
    print("简单修复对比图已保存为 dof_simple_fix_comparison.jpg")

if __name__ == "__main__":
    analyze_bright_line_issues()
    test_simple_fix()
    print("\n亮线问题分析和修复完成!")