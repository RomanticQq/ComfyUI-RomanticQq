import os
import cv2
import uuid
import time
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import json

class  TEXT_JSON:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"default": None}),
                "key": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("word", "word_part_of_speech")
    FUNCTION = "test"
    CATEGORY = "RomanticQq/text"
    def test(self, input, key):
        print("input: ", input)
        json_kv = json.loads(input)
        keys = key.split(",")
        return (json_kv[keys[0]],json_kv[keys[1]])