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

class LoraPath:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        lora_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split("custom_nodes")[0], "models", "loras")
        file_list = []
        for root, dirs, files in os.walk(lora_path):
            for file in files:
                file_path = os.path.join(root, file).split("models/loras/")[1]
                if file_path.startswith(".") or file_path.endswith((".json", ".txt", ".md", ".png", ".jpg", ".jpeg", ".sh", ".toml", ".zip", ".ini", ".bat", ".rar", ".tar", ".tar.gz")):
                    continue
                file_list.append(file_path)
        file_list.sort()
        return {
            "required": {
                "lora_path": (file_list,),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    def test(self, lora_path):
        lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)).split("custom_nodes")[0], "models", "loras")
        lora_all_path = os.path.join(lora_dir, lora_path)
        if not os.path.exists(lora_all_path):
            raise FileNotFoundError(f"Lora path '{lora_all_path}' does not exist.")
        print("lora_all_path:", lora_all_path)
        return (lora_all_path,)
