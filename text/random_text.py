import os
import cv2
import uuid
import time
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class  RANDOM_TEXT:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("LIST", {"default": None}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/text"
    def test(self, input, random_seed):
        texts = [t for t in input if t is not None and len(t) > 0]
        if len(texts) == 0:
            return ("",)
        np.random.seed(random_seed)
        text = np.random.choice(texts)
        return (text,)