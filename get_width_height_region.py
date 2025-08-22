import os
import cv2
import uuid
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class GetWidthHeightRegion:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "top": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "width": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    DESCRIPTION = "Get a region of the image based on specified coordinates and dimensions."
    def test(self, image, left, top, width, height):
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        img = image.numpy()[0]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_img_path, img)
        img = Image.open(tmp_img_path).convert("RGB")
        box = (left, top, width, height)
        region = img.crop(box)
        region.save(tmp_img_path)
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        return (img,)
    