#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 iOS App 图标：紫色渐变背景 + 白色轮廓的小人街舞造型。
输出：在 Assets.xcassets/AppIcon.appiconset 下生成各尺寸 PNG，并写入 Contents.json。
"""
import os
from math import sqrt
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "Assets.xcassets")
APPICON = os.path.join(ASSETS, "AppIcon.appiconset")

os.makedirs(APPICON, exist_ok=True)

# 颜色定义（紫色渐变：从左下角到右上角）
START = (139, 92, 246)   # 普通紫色 (#8B5CF6)
END   = (76, 29, 149)    # 深紫色 (#4C1D95)

def lerp(a, b, t):
    return int(a + (b - a) * t)

def make_gradient(w=1024, h=1024):
    img = Image.new("RGB", (w, h), (0, 0, 0))
    px = img.load()
    # 从左下角到右上角的渐变
    for y in range(h):
        for x in range(w):
            # 计算从左下角(0,h-1)到右上角(w-1,0)的距离比例
            # 左下角为起点(普通紫色)，右上角为终点(深紫色)
            diagonal_length = sqrt((w - 1) ** 2 + (h - 1) ** 2)
            current_distance = sqrt(x ** 2 + (h - 1 - y) ** 2)
            t = min(1.0, current_distance / diagonal_length)
            
            r = lerp(START[0], END[0], t)
            g = lerp(START[1], END[1], t)
            b = lerp(START[2], END[2], t)
            px[x, y] = (r, g, b)
    # 叠加一个柔和的径向亮斑（营造舞台灯光效果）
    cx, cy = int(w * 0.5), int(h * 0.35)
    radius = int(h * 0.55)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    opx = overlay.load()
    for y in range(h):
        for x in range(w):
            d = sqrt((x - cx) ** 2 + (y - cy) ** 2)
            t = max(0.0, 1.0 - d / radius)
            alpha = int(80 * (t ** 2))
            opx[x, y] = (255, 255, 255, alpha)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img

def draw_dancer(base: Image) -> Image:
    """在给定图像上绘制白色轮廓的人形（街舞姿态）。"""
    img = base.convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # 线宽与圆半径
    sw = max(6, int(w * 0.03))  # 约 30px for 1024
    head_r = int(w * 0.08)      # 约 82px

    # 关键点坐标（相对布局）
    cx = int(w * 0.5)
    head_cy = int(h * 0.28)
    torso_top = (cx, head_cy + head_r)
    torso_bottom_y = int(head_cy + head_r + h * 0.22)
    torso_bottom = (cx, torso_bottom_y)

    # 头部轮廓
    draw.ellipse([
        (cx - head_r, head_cy - head_r),
        (cx + head_r, head_cy + head_r)
    ], outline=(255, 255, 255), width=sw)

    # 躯干
    draw.line([torso_top, torso_bottom], fill=(255, 255, 255), width=sw)

    # 左臂（高举，含肘部）
    left_shoulder = (cx - int(w * 0.08), head_cy + int(h * 0.02))
    left_elbow = (cx - int(w * 0.22), head_cy - int(h * 0.02))
    left_hand = (cx - int(w * 0.28), head_cy - int(h * 0.10))
    draw.line([torso_top, left_shoulder, left_elbow, left_hand], fill=(255,255,255), width=sw, joint="curve")

    # 右臂（前伸，含肘部）
    right_shoulder = (cx + int(w * 0.08), head_cy + int(h * 0.01))
    right_elbow = (cx + int(w * 0.20), head_cy + int(h * 0.02))
    right_hand = (cx + int(w * 0.28), head_cy - int(h * 0.05))
    draw.line([torso_top, right_shoulder, right_elbow, right_hand], fill=(255,255,255), width=sw, joint="curve")

    # 左腿（弯曲）
    left_knee = (cx - int(w * 0.12), torso_bottom_y + int(h * 0.16))
    left_foot = (cx - int(w * 0.06), torso_bottom_y + int(h * 0.32))
    draw.line([torso_bottom, left_knee, left_foot], fill=(255,255,255), width=sw, joint="curve")

    # 右腿（伸展）
    right_knee = (cx + int(w * 0.16), torso_bottom_y + int(h * 0.12))
    right_foot = (cx + int(w * 0.26), torso_bottom_y + int(h * 0.30))
    draw.line([torso_bottom, right_knee, right_foot], fill=(255,255,255), width=sw, joint="curve")

    # 地面椭圆（影子/舞台效果）
    shadow_w = int(w * 0.55)
    shadow_h = int(h * 0.08)
    shadow_rect = [
        (cx - shadow_w // 2, torso_bottom_y + int(h * 0.26) - shadow_h // 2),
        (cx + shadow_w // 2, torso_bottom_y + int(h * 0.26) + shadow_h // 2)
    ]
    draw.ellipse(shadow_rect, outline=(255,255,255), width=max(4, sw//2))

    return img.convert("RGB")

# 生成基础 1024 图标
base = make_gradient(1024, 1024)
icon1024 = draw_dancer(base)
path1024 = os.path.join(APPICON, "Icon-App-1024x1024.png")
icon1024.save(path1024, format="PNG")
print("[OK] Base icon saved:", path1024)

# 需要的尺寸（文件名: (w, h)）
SIZES = {
    # iPhone
    "Icon-App-20x20@2x-iphone.png": (40, 40),
    "Icon-App-20x20@3x-iphone.png": (60, 60),
    "Icon-App-29x29@2x-iphone.png": (58, 58),
    "Icon-App-29x29@3x-iphone.png": (87, 87),
    "Icon-App-40x40@2x-iphone.png": (80, 80),
    "Icon-App-40x40@3x-iphone.png": (120, 120),
    "Icon-App-60x60@2x-iphone.png": (120, 120),
    "Icon-App-60x60@3x-iphone.png": (180, 180),
    # iPad
    "Icon-App-20x20@1x-ipad.png": (20, 20),
    "Icon-App-20x20@2x-ipad.png": (40, 40),
    "Icon-App-29x29@1x-ipad.png": (29, 29),
    "Icon-App-29x29@2x-ipad.png": (58, 58),
    "Icon-App-40x40@1x-ipad.png": (40, 40),
    "Icon-App-40x40@2x-ipad.png": (80, 80),
    "Icon-App-76x76@1x-ipad.png": (76, 76),
    "Icon-App-76x76@2x-ipad.png": (152, 152),
    "Icon-App-83.5x83.5@2x-ipad.png": (167, 167),
}

for name, (w, h) in SIZES.items():
    img = icon1024.resize((w, h), resample=Image.LANCZOS)
    out = os.path.join(APPICON, name)
    img.save(out, format="PNG")
    print("[OK]", name)

# 写入 Contents.json
contents = {
  "images": [
    # iPhone
    {"size": "20x20", "idiom": "iphone", "filename": "Icon-App-20x20@2x-iphone.png", "scale": "2x"},
    {"size": "20x20", "idiom": "iphone", "filename": "Icon-App-20x20@3x-iphone.png", "scale": "3x"},
    {"size": "29x29", "idiom": "iphone", "filename": "Icon-App-29x29@2x-iphone.png", "scale": "2x"},
    {"size": "29x29", "idiom": "iphone", "filename": "Icon-App-29x29@3x-iphone.png", "scale": "3x"},
    {"size": "40x40", "idiom": "iphone", "filename": "Icon-App-40x40@2x-iphone.png", "scale": "2x"},
    {"size": "40x40", "idiom": "iphone", "filename": "Icon-App-40x40@3x-iphone.png", "scale": "3x"},
    {"size": "60x60", "idiom": "iphone", "filename": "Icon-App-60x60@2x-iphone.png", "scale": "2x"},
    {"size": "60x60", "idiom": "iphone", "filename": "Icon-App-60x60@3x-iphone.png", "scale": "3x"},
    # iPad
    {"size": "20x20", "idiom": "ipad", "filename": "Icon-App-20x20@1x-ipad.png", "scale": "1x"},
    {"size": "20x20", "idiom": "ipad", "filename": "Icon-App-20x20@2x-ipad.png", "scale": "2x"},
    {"size": "29x29", "idiom": "ipad", "filename": "Icon-App-29x29@1x-ipad.png", "scale": "1x"},
    {"size": "29x29", "idiom": "ipad", "filename": "Icon-App-29x29@2x-ipad.png", "scale": "2x"},
    {"size": "40x40", "idiom": "ipad", "filename": "Icon-App-40x40@1x-ipad.png", "scale": "1x"},
    {"size": "40x40", "idiom": "ipad", "filename": "Icon-App-40x40@2x-ipad.png", "scale": "2x"},
    {"size": "76x76", "idiom": "ipad", "filename": "Icon-App-76x76@1x-ipad.png", "scale": "1x"},
    {"size": "76x76", "idiom": "ipad", "filename": "Icon-App-76x76@2x-ipad.png", "scale": "2x"},
    {"size": "83.5x83.5", "idiom": "ipad", "filename": "Icon-App-83.5x83.5@2x-ipad.png", "scale": "2x"},
    # App Store
    {"size": "1024x1024", "idiom": "ios-marketing", "filename": "Icon-App-1024x1024.png", "scale": "1x"}
  ],
  "info": {"version": 1, "author": "xcode"}
}

import json
with open(os.path.join(APPICON, "Contents.json"), "w", encoding="utf-8") as f:
    json.dump(contents, f, ensure_ascii=False, indent=2)

# 资产目录根 Contents.json（如果不存在）
ASSETS_CONTENTS = os.path.join(ASSETS, "Contents.json")
if not os.path.exists(ASSETS_CONTENTS):
    with open(ASSETS_CONTENTS, "w", encoding="utf-8") as f:
        json.dump({"info": {"version": 1, "author": "xcode"}}, f, ensure_ascii=False, indent=2)

print("[DONE] 所有图标与 Contents.json 已生成到:", APPICON)