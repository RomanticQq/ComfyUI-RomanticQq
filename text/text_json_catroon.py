import os
import cv2
import uuid
import time
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import json

class  TEXT_JSON_CATROON:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"default": None}),
                "key1": ("STRING", {"default": None}),
                "key2": ("STRING", {"default": None}),
                "key3": ("STRING", {"default": None}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("value1", "value2", "value3")
    FUNCTION = "test"
    CATEGORY = "RomanticQq/text"
    def test(self, input, key1, key2, key3):
        value1 = None
        value2 = None
        value3 = None
        print("input: ", input)
        json_kv = json.loads(input.replace("```json", "").replace("```", ""))
        if key1 is not None and len(key1) > 0:
            value1 = json_kv[key1]
        if key2 is not None and len(key2) > 0:
            value2 = json_kv[key2]
        if key3 is not None and len(key3) > 0:
            value3 = json_kv[key3]
        return (value1, value2, value3)

