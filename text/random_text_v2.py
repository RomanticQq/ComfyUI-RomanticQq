import os
import cv2
import uuid
import time
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class  RANDOM_TEXT_V2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"default": None}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/text"
    def test(self, input, random_seed):
        np.random.seed(random_seed)
        input = input.split("\n")
        texts = [t for t in input if t is not None and len(t) > 0]
        if len(texts) == 0:
            return ("",)
        
        text = texts[np.random.randint(0, len(texts))]
        return (text,)