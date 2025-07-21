import os
import cv2
import uuid
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class AddText:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "word": ("STRING", {"default": ''}),
                "english_sentence": ("STRING", {"default": ''}),
                "chinese_sentence": ("STRING", {"default": ''}),
                "font_color": ("STRING", {"default": "black"}),
                "resize": ("BOOLEAN", {"default": True}),
                "width": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 2048, "step": 1}),
                "word_font_size": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "english_sentence_font_size": ("INT", {"default":30, "min": 0, "max": 100, "step": 1}),
                "chinese_sentence_font_size": ("INT", {"default":30, "min": 0, "max": 100, "step": 1}),
                "top": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1}),
                "bottom": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "gap_english_chinese_sentence": ("INT", {"default": 20, "min": 0, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    def test(self, image, word, english_sentence, chinese_sentence, font_color, resize, width, height, word_font_size=50, english_sentence_font_size=30, chinese_sentence_font_size=30, top=30, bottom=50, gap_english_chinese_sentence=20):
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        img = image.numpy()[0]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_img_path, img)
        image = Image.open(tmp_img_path).convert("RGB")
        if resize:
            image = image.resize((width, height), Image.LANCZOS)
        draw = ImageDraw.Draw(image)
        word_font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "font/微软雅黑粗体.TTC"), word_font_size)
        word_bbox = draw.textbbox((0, 0), word, font=word_font)
        word_text_width = word_bbox[2] - word_bbox[0]
        word_text_height = word_bbox[3] - word_bbox[1]
        word_x_position = (image.width - word_text_width) // 2
        word_y_position = top
        draw.text((word_x_position, word_y_position), word, font=word_font, fill=font_color)


        chinese_sentence_font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "font/微软雅黑常规.TTC"), chinese_sentence_font_size)
        chinese_sentence_bbox = draw.textbbox((0, 0), chinese_sentence, font=chinese_sentence_font)
        chinese_sentence_text_width = chinese_sentence_bbox[2] - chinese_sentence_bbox[0]
        chinese_sentence_text_height = chinese_sentence_bbox[3] - chinese_sentence_bbox[1]
        chinese_sentence_x_position = (image.width - chinese_sentence_text_width) // 2
        chinese_sentence_y_position = image.height - chinese_sentence_text_height - bottom
        draw.text((chinese_sentence_x_position, chinese_sentence_y_position), chinese_sentence, font=chinese_sentence_font, fill=font_color)


        english_sentence_font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "font/微软雅黑粗体.TTC"), english_sentence_font_size)
        english_sentence_bbox = draw.textbbox((0, 0), english_sentence, font=english_sentence_font)
        english_sentence_text_width = english_sentence_bbox[2] - english_sentence_bbox[0]
        english_sentence_text_height = english_sentence_bbox[3] - english_sentence_bbox[1]
        english_sentence_x_position = (image.width - english_sentence_text_width) // 2
        english_sentence_y_position = chinese_sentence_y_position - english_sentence_text_height - gap_english_chinese_sentence
        draw.text((english_sentence_x_position, english_sentence_y_position), english_sentence, font=english_sentence_font, fill=font_color)
        image.save(tmp_img_path)
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        return (img,)