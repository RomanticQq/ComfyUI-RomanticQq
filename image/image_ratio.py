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
from openai import OpenAI
class ImageRatio:
    def __init__(self):
        self.tmp_dir = {"2:3":[1024,1536], "3:4":[1536,2048],"1:1":[1024,1024],"4:3":[2048,1536], "3:2":[1536,1024], "16:9":[1920,1080], "21:9":[2560,1080]}
    @classmethod
    def INPUT_TYPES(s):
        aspectRatio_list = ['2:3', '3:4', '1:1', '4:3', '3:2', '16:9', '21:9']
        return {
            "required": {
                "aspectRatio": (aspectRatio_list,),
            },
        }

    RETURN_TYPES = ("INT", "INT","FLOAT")
    RETURN_NAMES = ("width", "height","ratio")
    FUNCTION = "test"
    CATEGORY = "RomanticQq/image"
    def test(self, aspectRatio):
        width, height = self.tmp_dir[aspectRatio]
        return (width, height, width/height)