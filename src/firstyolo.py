# 修复导入问题和迭代None类型的问题
from ultralytics import YOLO  # type: ignore
import os

model = YOLO("yolo11x.pt")  # 使用项目中的模型文件

# 明确指定保存路径和项目名称
results = model.predict("market-square.mp4", save=True, project="runs", name="detect", exist_ok=True)
print(results)
print("===============")    

# 打印保存路径
save_dir = os.path.join("runs", "detect")
print(f"视频保存路径: {os.path.abspath(save_dir)}")

# 检查results是否为空以及results[0].boxes是否存在
if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
    for bbox in results[0].boxes:
        print(bbox)
else:
    print("未检测到任何边界框或结果为空")