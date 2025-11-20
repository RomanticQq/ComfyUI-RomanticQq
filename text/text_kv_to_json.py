import os
import cv2
import uuid
import time
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import json

class  TEXT_KV_TO_JSON:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key1": ("STRING", {"default": None}),
                "value1": ("STRING", {"default": None}),
                "key2": ("STRING", {"default": None}),
                "value2": ("STRING", {"default": None}),
                "key3": ("STRING", {"default": None}),
                "value3": ("STRING", {"default": None}),
                "key4": ("STRING", {"default": None}),
                "value4": ("STRING", {"default": None}),
                "key5": ("STRING", {"default": None}),
                "value5": ("STRING", {"default": None}),
                "key6": ("STRING", {"default": None}),
                "value6": ("STRING", {"default": None}),
                "key7": ("STRING", {"default": None}),
                "value7": ("STRING", {"default": None}),
                "key8": ("STRING", {"default": None}),
                "value8": ("STRING", {"default": None}),
                "key9": ("STRING", {"default": None}),
                "value9": ("STRING", {"default": None}),
                "key10": ("STRING", {"default": None}),
                "value10": ("STRING", {"default": None}),
                "key11": ("STRING", {"default": None}),
                "value11": ("STRING", {"default": None}),
                "key12": ("STRING", {"default": None}),
                "value12": ("STRING", {"default": None}),
                "key13": ("STRING", {"default": None}),
                "value13": ("STRING", {"default": None}),
                "key14": ("STRING", {"default": None}),
                "value14": ("STRING", {"default": None}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/text"
    def test(self, key1, value1, key2, value2, key3, value3, key4, value4, key5, value5, key6, value6, key7, value7, key8, value8, key9, value9, key10, value10, key11, value11, key12, value12, key13, value13, key14, value14):
        json_kv = {}
        if key1 is not None and len(key1) > 0:
            json_kv[key1] = value1
        if key2 is not None and len(key2) > 0:
            json_kv[key2] = value2
        if key3 is not None and len(key3) > 0:
            json_kv[key3] = value3
        if key4 is not None and len(key4) > 0:
            json_kv[key4] = value4
        if key5 is not None and len(key5) > 0:
            json_kv[key5] = value5
        if key6 is not None and len(key6) > 0:
            json_kv[key6] = value6
        if key7 is not None and len(key7) > 0:
            json_kv[key7] = value7
        if key8 is not None and len(key8) > 0:
            json_kv[key8] = value8
        if key9 is not None and len(key9) > 0:
            json_kv[key9] = value9
        if key10 is not None and len(key10) > 0:
            json_kv[key10] = value10
        if key11 is not None and len(key11) > 0:
            json_kv[key11] = value11
        if key12 is not None and len(key12) > 0:
            json_kv[key12] = value12
        if key13 is not None and len(key13) > 0:
            json_kv[key13] = value13
        if key14 is not None and len(key14) > 0:
            json_kv[key14] = value14
        res = json.dumps(json_kv, indent=4, ensure_ascii=False)
        return (res,)