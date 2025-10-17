#!/usr/bin/env python3
"""
景深效果优化演示
展示优化前后的对比效果
"""

import cv2
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

def create_optimization_demo():
    """创建优化演示"""
    print("=== 景深效果优化演示 ===")
    
    # 读取不同版本的结果
    original = cv2.imread("sample_bus.jpg")
    basic_result = cv2.imread("dof_test_result.jpg")
    improved_result = cv2.imread("dof_final_result.jpg")
    
    if original is None or basic_result is None or improved_result is None:
        print("错误: 无法读取必要的图像文件")
        return
    
    height, width = original.shape[:2]
    
    # 创建三联对比图
    comparison_width = width * 3
    comparison_height = height + 100  # 为标题留出空间
    
    comparison = np.zeros((comparison_height, comparison_width, 3), dtype=np.uint8)
    
    # 放置图像
    comparison[50:50+height, :width] = original
    comparison[50:50+height, width:2*width] = basic_result
    comparison[50:50+height, 2*width:3*width] = improved_result
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "原始图像", (width//2-80, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "基础景深效果", (width + width//2-100, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "优化景深效果", (2*width + width//2-100, 30), font, 1, (0, 255, 0), 2)
    
    # 添加说明
    cv2.putText(comparison, "边缘平滑优化对比", (20, comparison_height-20), font, 0.7, (255, 255, 255), 1)
    
    cv2.imwrite("dof_optimization_demo.jpg", comparison)
    print("优化演示图已保存为 dof_optimization_demo.jpg")
    
    # 创建局部放大对比图以突出边缘改进
    if width > 400 and height > 400:
        # 选择一个包含清晰边缘的区域
        crop_x = width - 350
        crop_y = 100
        crop_size = 300
        
        if crop_x + crop_size <= width and crop_y + crop_size <= height:
            # 提取局部区域
            original_crop = original[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
            basic_crop = basic_result[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
            improved_crop = improved_result[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
            
            # 放大2倍
            zoom_factor = 2
            original_zoom = cv2.resize(original_crop, 
                                     (crop_size * zoom_factor, crop_size * zoom_factor), 
                                     interpolation=cv2.INTER_LINEAR)
            basic_zoom = cv2.resize(basic_crop, 
                                  (crop_size * zoom_factor, crop_size * zoom_factor), 
                                  interpolation=cv2.INTER_LINEAR)
            improved_zoom = cv2.resize(improved_crop, 
                                     (crop_size * zoom_factor, crop_size * zoom_factor), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # 创建放大对比图
            zoom_width = crop_size * zoom_factor * 3
            zoom_height = crop_size * zoom_factor + 80
            
            zoom_comparison = np.zeros((zoom_height, zoom_width, 3), dtype=np.uint8)
            zoom_comparison[50:50+crop_size * zoom_factor, :crop_size * zoom_factor] = original_zoom
            zoom_comparison[50:50+crop_size * zoom_factor, crop_size * zoom_factor:2 * crop_size * zoom_factor] = basic_zoom
            zoom_comparison[50:50+crop_size * zoom_factor, 2 * crop_size * zoom_factor:] = improved_zoom
            
            # 添加标签
            cv2.putText(zoom_comparison, "原始", (crop_size * zoom_factor//2-50, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(zoom_comparison, "基础效果", (crop_size * zoom_factor + crop_size * zoom_factor//2-70, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(zoom_comparison, "优化效果", (2 * crop_size * zoom_factor + crop_size * zoom_factor//2-70, 30), font, 0.8, (0, 255, 0), 2)
            
            cv2.imwrite("dof_edge_improvement_zoom.jpg", zoom_comparison)
            print("边缘改进放大对比图已保存为 dof_edge_improvement_zoom.jpg")
    
    # 创建边缘分析图
    create_edge_analysis(original, basic_result, improved_result)

def create_edge_analysis(original, basic_result, improved_result):
    """创建边缘分析图"""
    print("\n创建边缘分析...")
    
    height, width = original.shape[:2]
    
    # 选择一个代表性区域进行分析
    analysis_x, analysis_y, analysis_size = 200, 150, 200
    
    if analysis_x + analysis_size <= width and analysis_y + analysis_size <= height:
        # 提取分析区域
        orig_region = original[analysis_y:analysis_y+analysis_size, analysis_x:analysis_x+analysis_size]
        basic_region = basic_result[analysis_y:analysis_y+analysis_size, analysis_x:analysis_x+analysis_size]
        improved_region = improved_result[analysis_y:analysis_y+analysis_size, analysis_x:analysis_x+analysis_size]
        
        # 应用Canny边缘检测来比较边缘质量
        orig_edges = cv2.Canny(orig_region, 50, 150)
        basic_edges = cv2.Canny(basic_region, 50, 150)
        improved_edges = cv2.Canny(improved_region, 50, 150)
        
        # 创建边缘对比图
        edge_comparison = np.zeros((analysis_size * 2, analysis_size * 3, 3), dtype=np.uint8)
        
        # 第一行：原始图像和边缘
        edge_comparison[:analysis_size, :analysis_size] = orig_region
        orig_edges_3channel = cv2.merge([orig_edges, orig_edges, orig_edges])
        edge_comparison[:analysis_size, analysis_size:2*analysis_size] = orig_edges_3channel
        edge_comparison[:analysis_size, 2*analysis_size:3*analysis_size] = orig_edges_3channel
        
        # 第二行：基础效果边缘和优化效果边缘
        basic_edges_3channel = cv2.merge([basic_edges, basic_edges, basic_edges])
        edge_comparison[analysis_size:2*analysis_size, :analysis_size] = basic_edges_3channel
        
        improved_edges_3channel = cv2.merge([improved_edges, improved_edges, improved_edges])
        edge_comparison[analysis_size:2*analysis_size, analysis_size:2*analysis_size] = improved_edges_3channel
        
        # 创建差异图（突出改进）
        diff_edges = cv2.absdiff(basic_edges, improved_edges)
        diff_edges_3channel = cv2.merge([diff_edges, diff_edges, diff_edges])
        edge_comparison[analysis_size:2*analysis_size, 2*analysis_size:3*analysis_size] = diff_edges_3channel
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(edge_comparison, "原始图像", (10, 30), font, 0.6, (0, 255, 0), 1)
        cv2.putText(edge_comparison, "原始边缘", (analysis_size + 10, 30), font, 0.6, (0, 255, 0), 1)
        cv2.putText(edge_comparison, "基础边缘", (10, analysis_size + 30), font, 0.6, (0, 255, 0), 1)
        cv2.putText(edge_comparison, "优化边缘", (analysis_size + 10, analysis_size + 30), font, 0.6, (0, 255, 0), 1)
        cv2.putText(edge_comparison, "改进差异", (2*analysis_size + 10, analysis_size + 30), font, 0.6, (0, 255, 0), 1)
        
        cv2.imwrite("dof_edge_analysis.jpg", edge_comparison)
        print("边缘分析图已保存为 dof_edge_analysis.jpg")

if __name__ == "__main__":
    create_optimization_demo()
    print("\n优化演示完成!")