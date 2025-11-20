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
                "key4": ("STRING", {"default": None}),
                "key5": ("STRING", {"default": None}),
                "key6": ("STRING", {"default": None}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("value1", "value2", "value3", "value4", "value5", "value6")
    FUNCTION = "test"
    CATEGORY = "RomanticQq/text"
    def test(self, input, key1, key2, key3, key4, key5, key6):
        value1 = None
        value2 = None
        value3 = None
        value4 = None
        value5 = None
        value6 = None
        print("input: ", input)
        json_kv = json.loads(input.replace("```json", "").replace("```", ""))
        if key1 is not None and len(key1) > 0:
            value1 = json_kv[key1]
        if key2 is not None and len(key2) > 0:
            value2 = json_kv[key2]
        if key3 is not None and len(key3) > 0:
            value3 = json_kv[key3]
        if key4 is not None and len(key4) > 0:
            value4 = json_kv[key4]
        if key5 is not None and len(key5) > 0:
            value5 = json_kv[key5]
        if key6 is not None and len(key6) > 0:
            value6 = json_kv[key6]
        return (value1, value2, value3, value4, value5, value6)

