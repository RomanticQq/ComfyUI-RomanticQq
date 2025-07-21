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

class AddTwoImage:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "small_image": ("IMAGE",),
                "radius": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "large_image": ("IMAGE", {"default": None}),
                "random_background": ("BOOLEAN", {"default": False}),
                "background_rgb": ("STRING", {"default": "255,255,255"}),
                "padding_size": ("INT", {"default": None}),
                "width": ("INT", {"default": None}),
                "height": ("INT", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    def test(self, small_image, radius, large_image=None, random_background=False, background_rgb="255,255,255", padding_size=None, width=None, height=None):
        small_tmp_img_name = str(uuid.uuid4()) + ".jpg"
        small_tmp_img_path = os.path.join(self.tmp_dir, small_tmp_img_name)
        small_img = small_image.numpy()[0]
        small_img = (small_img * 255).astype(np.uint8)
        small_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(small_tmp_img_path, small_img)
        if padding_size is not None:
            width, height = Image.open(small_tmp_img_path).size
            width = width + padding_size
            height = height + padding_size
        large_tmp_img_name = str(uuid.uuid4()) + ".jpg"
        large_tmp_img_path = os.path.join(self.tmp_dir, large_tmp_img_name)
        if large_image is not None:
            large_img = large_image.numpy()[0]
            large_img = (large_img * 255).astype(np.uint8)
            large_img = cv2.cvtColor(large_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(large_tmp_img_path, large_img)
        elif random_background:
            color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
            image_array = np.full((height, width, 3), color, dtype=np.uint8)
            image = Image.fromarray(image_array)
            image.save(large_tmp_img_path)
        elif background_rgb:
            # 创建一个随机背景的大图
            background_rgb = tuple(map(int, background_rgb.split(',')))
            large_img = Image.new("RGB", (width, height), background_rgb)
            large_img.save(large_tmp_img_path)
        else:
            exit("请提供大图或随机背景参数")

        # 读取大图和小图
        large_img = Image.open(large_tmp_img_path)
        small_img = Image.open(small_tmp_img_path)

        # 获取大图和小图的尺寸
        large_width, large_height = large_img.size
        small_width, small_height = small_img.size

        # 计算小图放置的位置，确保在大图的正中心
        x_offset = (large_width - small_width) // 2
        y_offset = (large_height - small_height) // 2

        # 为小图创建圆角边框
        radius = 40  # 圆角半径
        small_img = small_img.convert("RGBA")
        mask = Image.new("L", (small_width, small_height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, small_width, small_height), radius, fill=255)
        small_img.putalpha(mask)

        # 在大图上粘贴小图
        large_img.paste(small_img, (x_offset, y_offset), small_img)

        # 保存最终合成的图片
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        large_img.save(tmp_img_path)
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        os.remove(small_tmp_img_path)
        os.remove(large_tmp_img_path)
        return (img,)