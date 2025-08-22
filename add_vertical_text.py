import os
import cv2
import uuid
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class ADD_VERTICAL_TEXT:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
    @classmethod
    def INPUT_TYPES(s):
        font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"font")
        font_files = [f for f in os.listdir(font_dir)]
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": ''}),
                "font": (font_files,),
                "font_color": ("STRING", {"default": "black"}),
                "font_size": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "x": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "y": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "position": (["auto", "left", "right"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    DESCRIPTION = "Add vertical text to an image at specified coordinates."
    def test(self, image, text, font, font_color, font_size, x, y, position):
        if len(text) == 0:
            return (image,)
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        img = image.numpy()[0]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_img_path, img)
        image = Image.open(tmp_img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),"font"), font), font_size)
        
        if len(text) > 0:
            char = text[0]
            img_h, img_w = image.size
            bbox = draw.textbbox((x, y), char, font=font)
            text_width = bbox[2] - bbox[0]
            positions = [x, img_w-text_width-x]
            if position == "left":
                x = positions[0]
            elif position == "right":
                x = positions[1]
            else:
                x = np.random.choice(positions)

        for char in text:
            # 获取字符的边界框
            bbox = draw.textbbox((x, y), char, font=font)
            text_height = bbox[3] - bbox[1]
            draw.text((x, y), char, font=font, fill=font_color)
            y += text_height

        # 保存修改后的图片
        image.save(tmp_img_path)
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        return (img,)