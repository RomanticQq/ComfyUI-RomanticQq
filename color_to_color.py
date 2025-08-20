import os
import cv2
import uuid
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import numpy as np
from matplotlib import colors as mcolors
import colorsys


# 猴子补丁修复 numpy.asscalar 问题
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()

def hex_to_lab(hex_color):
    """将十六进制颜色转换为Lab颜色空间"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:  # 处理简写格式如 #FFF
        hex_color = ''.join([c*2 for c in hex_color])
    rgb = sRGBColor.new_from_rgb_hex(hex_color)
    return convert_color(rgb, LabColor)

def find_closest_color_delta_e(new_hex, known_hex_colors):
    """使用Delta E方法找出最接近的颜色"""
    new_lab = hex_to_lab(new_hex)
    min_distance = float('inf')
    closest_color = None
    
    for known_hex in known_hex_colors:
        try:
            known_lab = hex_to_lab(known_hex)
            dist = delta_e_cie2000(new_lab, known_lab)
            if dist < min_distance:
                min_distance = dist
                closest_color = known_hex
        except ValueError as e:
            print(f"跳过无效颜色 {known_hex}: {str(e)}")
            continue
    
    return min_distance, closest_color  # 返回距离值更有参考意义


def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB元组"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:  # 处理简写格式如 #FFF
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def hex_to_hsv(hex_color):
    """将十六进制颜色转换为HSV"""
    rgb = hex_to_rgb(hex_color)
    return colorsys.rgb_to_hsv(*[x/255.0 for x in rgb])

def hsv_distance(color1, color2):
    """计算HSV空间的距离（加权）"""
    h1, s1, v1 = color1
    h2, s2, v2 = color2
    dh = min(abs(h1-h2), 1-abs(h1-h2)) * 2  # 色调是环形的
    ds = abs(s1-s2)
    dv = abs(v1-v2)
    return (dh*0.6 + ds*0.2 + dv*0.2)  # 给色调更高权重

def find_closest_color_hsv(new_hex, known_hex_colors):
    """使用HSV空间找出最接近的颜色"""
    new_hsv = hex_to_hsv(new_hex)
    distances = []
    
    for known_hex in known_hex_colors:
        known_hsv = hex_to_hsv(known_hex)
        dist = hsv_distance(new_hsv, known_hsv)
        distances.append((dist, known_hex))
    
    distances.sort()
    return distances[0]

def color_distance(color1, color2):
    """计算两个RGB颜色之间的欧氏距离"""
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return ((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)**0.5

def find_closest_color(new_hex, known_hex_colors):
    """找出已知颜色中最接近新颜色的颜色"""
    new_rgb = hex_to_rgb(new_hex)
    distances = []
    
    for known_hex in known_hex_colors:
        known_rgb = hex_to_rgb(known_hex)
        dist = color_distance(new_rgb, known_rgb)
        distances.append((dist, known_hex))
    
    # 按距离排序并返回最接近的颜色
    distances.sort()
    return distances[0]

class ColorToColor:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_palettes": ("LIST",),
                "mcolors_name": (['CSS4_COLORS', 'XKCD_COLORS', 'TABLEAU_COLORS'],),
                "compare_method": (['delta_e', 'hsv', 'rgb'],),
                "topk": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST", "STRING", "STRING")
    RETURN_NAMES = ("color_list", "color_name_list", "color_distance", "color_name_str", "color_name_str_index")
    FUNCTION = "test"
    CATEGORY = "RomanticQq/COLOR"
    DESCRIPTION = "将颜色转换为已知颜色集中的最接近颜色, 支持CSS4、XKCD和Tableau颜色集，支持多种比较方法"
    def test(self, color_palettes, mcolors_name, compare_method, topk=None, index=None):
        color_palettes = color_palettes[0].split('\n')
        if mcolors_name == "CSS4_COLORS":
            colors_kv = mcolors.CSS4_COLORS
        elif mcolors_name == "XKCD_COLORS":
            colors_kv = mcolors.XKCD_COLORS
        elif mcolors_name == "TABLEAU_COLORS":
            colors_kv = mcolors.TABLEAU_COLORS
        else:
            raise ValueError("Invalid color set selected")
        colors_vk = {v:k for k, v in colors_kv.items()}
        known_colors = [v for k, v in colors_kv.items()]
        
        if compare_method == "delta_e":
            find_closest_color_fun = find_closest_color_delta_e
        elif compare_method == "hsv":
            find_closest_color_fun = find_closest_color_hsv
        elif compare_method == "rgb":
            find_closest_color_fun = find_closest_color
        
        color_list = []
        distance_list = []
        color_name_list = []
        for color in color_palettes:
            color = color.upper()
            distance, closest = find_closest_color_fun(color, known_colors)
            if mcolors_name == "XKCD_COLORS":
                color_name_list.append(colors_vk[closest].split(':')[1])
                print(f"使用 {compare_method} 方法最接近 {color} 颜色是 {colors_vk[closest].split(':')[1]}, 十六进制是 {closest}, 距离 {distance:.4f}")
            else:
                color_name_list.append(colors_vk[closest])
                print(f"使用 {compare_method} 方法最接近 {color} 颜色是 {colors_vk[closest]}, 十六进制是 {closest}, 距离 {distance:.4f}")
            color_list.append(closest)
            distance_list.append(distance)
        if index is not None and index < len(color_list):
            if index > len(color_name_list) - 1:
                index = len(color_name_list) - 1
            color_name_str_index = color_name_list[index]
        else:
            color_name_str_index = ''
        if topk is not None and topk > 0:
            if topk > len(color_list):
                topk = len(color_list)
            # 截断到 topk
            color_list = color_list[:topk]
            color_name_list = color_name_list[:topk]
            distance_list = distance_list[:topk]
        color_name_str = ', '.join(color_name_list)

        return (color_list, color_name_list, distance_list, color_name_str, color_name_str_index)