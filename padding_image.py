import os
import cv2
import uuid
import torch
import numpy as np
import os
import json
import requests
from minio import Minio
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageOps

class PaddingImage:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", {"default": 40, "min": 0, "max": 2048, "step": 1}),
                "color": ("STRING", {"default": "#FFFFFF"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    def test(self, image, size, color):
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        img = image.numpy()[0]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_img_path, img)



        image = Image.open(tmp_img_path).convert("RGB")
        original_width, original_height = image.size
        print(f"原图尺寸: {original_width}×{original_height}") 
        # 计算需要填充的尺寸
        size = 1024
        
        # 计算左右和上下的填充量
        pad_width = max(0, size - original_width)
        pad_height = max(0, size - original_height)
        
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        
        print(f"左右填充: {pad_left} + {pad_right} = {pad_width} 像素")
        print(f"上下填充: {pad_top} + {pad_bottom} = {pad_height} 像素")
        
        # 创建新图像并填充
        new_image = Image.new('RGB', (size, size), color)
        
        # 计算原图在新图像中的位置（居中）
        paste_x = pad_left
        paste_y = pad_top
        
        # 粘贴原图
        new_image.paste(image, (paste_x, paste_y))
        
        # 保存结果
        new_image.save(tmp_img_path)
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        return (img,)