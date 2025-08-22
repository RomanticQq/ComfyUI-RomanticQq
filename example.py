import os
import cv2
import uuid
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class AAA:
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
        print(type(image)) # <class 'torch.Tensor'>
        print(image.shape) # torch.Size([1, 1528, 1080, 3]) n,h,w,c
        img = image.numpy()[0]
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        box = (left, top, left + width, top + height)
        region = img.crop(box)
        region = np.array(region).astype(np.float32) / 255.0
        img = torch.from_numpy(region).unsqueeze(0)
        return (img,)
    