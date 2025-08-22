import os
import cv2
import uuid
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class ColorToImage:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": ("STRING",),
                "width": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/COLOR"
    def test(self, color, width, height):
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
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        return (img,)