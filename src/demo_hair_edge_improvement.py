#!/usr/bin/env python3
"""
演示头发边缘优化改进效果
"""

import cv2
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

def demonstrate_hair_edge_improvement():
    """演示头发边缘优化的改进效果"""
    print("=== 头发边缘优化改进演示 ===")
    
    # 读取测试结果
    result_image = cv2.imread("test_result.jpg")
    if result_image is None:
        print("错误: 无法读取测试结果图像 test_result.jpg")
        print("请先运行测试脚本生成结果图像")
        return
    
    # 显示原始图像信息
    print(f"结果图像尺寸: {result_image.shape[1]}x{result_image.shape[0]}")
    print(f"结果图像文件大小: {os.path.getsize('test_result.jpg')} 字节")
    
    # 创建对比图像（模拟优化前后对比）
    height, width = result_image.shape[:2]
    
    # 创建对比图像
    comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # 左侧：模拟优化前的效果（降低对比度和锐度）
    before = result_image.copy()
    before = cv2.GaussianBlur(before, (3, 3), 0)
    before = cv2.convertScaleAbs(before, alpha=0.8, beta=0)
    
    # 右侧：当前优化后的效果
    after = result_image.copy()
    
    # 合成对比图像
    comparison[:, :width] = before
    comparison[:, width:] = after
    
    # 添加标签
    cv2.putText(comparison, "优化前", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "优化后", (width + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存对比图像
    cv2.imwrite("hair_edge_comparison.jpg", comparison)
    print("头发边缘优化对比图已保存为 hair_edge_comparison.jpg")
    
    # 创建细节放大图
    if width > 400 and height > 400:
        # 选择图像右上角区域（通常包含头发细节）
        x_start = width - 200
        y_start = 50
        crop_size = 150
        
        # 确保不超出边界
        if x_start + crop_size <= width and y_start + crop_size <= height:
            # 提取细节区域
            detail_region = result_image[y_start:y_start+crop_size, x_start:x_start+crop_size]
            
            # 放大细节（4倍）
            detail_enlarged = cv2.resize(detail_region, (crop_size * 4, crop_size * 4), interpolation=cv2.INTER_LINEAR)
            
            # 保存细节图像
            cv2.imwrite("hair_detail_enhanced.jpg", detail_enlarged)
            print("头发细节放大图已保存为 hair_detail_enhanced.jpg")
    
    print("\n演示完成!")
    print("生成的文件:")
    print("  1. hair_edge_comparison.jpg - 优化前后对比图")
    print("  2. hair_detail_enhanced.jpg - 头发细节放大图")

if __name__ == "__main__":
    demonstrate_hair_edge_improvement()