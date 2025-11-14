import os
import cv2
import uuid
import torch
import numpy as np
import os
import json
import requests
from minio import Minio
from datetime import datetime, timedelta
import re
class Florence2GetWord:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/text"
    def test(self, text):
        cleaned = re.sub(r'<loc_\d+>', ' ', text)
        words = [w.strip() for w in cleaned.split() if w.strip()] 
        res_text = ','.join(words)
        return (res_text,)
