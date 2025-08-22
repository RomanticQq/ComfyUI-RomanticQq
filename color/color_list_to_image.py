import os
import cv2
import uuid
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class ColorListToImage:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_list": ("LIST",),
                "width": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 1}),
                "mode": (["horizontal", "vertical"], {"default": "horizontal"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/COLOR"
    def test(self, color_list, width, height, mode):
        tmp_img_path_list = []
        for color in color_list:
            tmp_img_name = str(uuid.uuid4()) + ".png"
            tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)

            # 创建一个图形
            print(f"Creating image with color: {color}, width: {width}, height: {height}")
            if not color.startswith('#'):
                color = '#' + color
        
            # 创建纯色图像
            img = Image.new('RGB', (width, height), color)
            # 保存图片到文件
            img.save(tmp_img_path)
            tmp_img_path_list.append(tmp_img_path)

        
        images = [Image.open(img_path) for img_path in tmp_img_path_list]
    
        # 获取单张图片的尺寸（假设所有图片尺寸相同）
        width, height = images[0].size
        
        # 创建新画布（宽度 = 单张宽度 × 图片数量，高度 = 单张高度）
        total_width = width * len(images)
        new_image = Image.new('RGB', (total_width, height))
        
        # 拼接图片
        x_offset = 0
        for img in images:
            new_image.paste(img, (x_offset, 0))
            x_offset += width
    
        # 保存结果
        new_image.save(tmp_img_path)
        img = cv2.imread(tmp_img_path)
        if mode == "vertical":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        for tmp_img_path in tmp_img_path_list:
            os.remove(tmp_img_path)
        return (img,)