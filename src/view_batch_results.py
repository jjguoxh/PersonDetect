#!/usr/bin/env python3
"""
查看批量处理结果的脚本
"""

import os
import glob

def view_batch_results():
    """
    查看批量处理结果
    """
    output_dir = "runs/batch_processed"
    
    if not os.path.exists(output_dir):
        print(f"输出目录 {output_dir} 不存在")
        return
        
    # 查找所有处理后的视频文件
    result_pattern = os.path.join(output_dir, "*_result.mp4")
    result_files = glob.glob(result_pattern)
    
    if not result_files:
        print(f"在目录 {output_dir} 中未找到处理后的视频文件")
        return
        
    print(f"找到 {len(result_files)} 个处理后的视频文件:")
    print("=" * 60)
    
    total_size = 0
    for i, result_file in enumerate(sorted(result_files), 1):
        file_size = os.path.getsize(result_file)
        total_size += file_size
        file_size_mb = file_size / (1024 * 1024)  # 转换为MB
        print(f"{i:2d}. {os.path.basename(result_file)}")
        print(f"    大小: {file_size_mb:.2f} MB")
        print(f"    路径: {result_file}")
        print()
        
    total_size_mb = total_size / (1024 * 1024)
    print(f"总处理文件数: {len(result_files)}")
    print(f"总大小: {total_size_mb:.2f} MB")
    print("=" * 60)
    
    print("\n处理完成的视频文件已保存在:")
    print(f"  {os.path.abspath(output_dir)}")
    print("\n您可以使用视频播放器查看这些文件。")

def main():
    print("查看批量处理结果...")
    view_batch_results()

if __name__ == "__main__":
    main()