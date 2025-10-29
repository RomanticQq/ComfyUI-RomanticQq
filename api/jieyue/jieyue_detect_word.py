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

class JIeyueDetectWordAPI:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.url = 'http://test-api.aiedevice.com/dictionary/device/recite_word/identify_image_word'  # 替换为实际接口URL
        self.keys = json.load(open(os.path.join(os.path.dirname(__file__).split('/api')[0], "keys.json"), "r"))
        self.headers = {
            "appId": self.keys["jiyue_detect_word"]["appId"],
            "babyId": self.keys["jiyue_detect_word"]["babyId"],
            "clientId": self.keys["jiyue_detect_word"]["clientId"],
            "RC-DEVICE-SESSION": self.keys["jiyue_detect_word"]["RC-DEVICE-SESSION"],
            "Content-Type": "application/json"
        }
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "imageUrl": ("STRING",{"default": None}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/api/jieyue"
    def test(self, imageUrl, random_seed=0):
        np.random.seed(random_seed)
        # model_id 对应模型


        print(os.getcwd())
        # Send the image to the server
        for i in range(3):
            print(f"第{i+1}次请求")
            try:
                data = {
                    "appId": self.keys["jiyue_detect_word"]["appId"],
                    "babyId": self.keys["jiyue_detect_word"]["babyId"],
                    "clientId": self.keys["jiyue_detect_word"]["clientId"],
                    "imageUrl": imageUrl
                }


                response = requests.post(self.url, headers=self.headers, json=data, timeout=60)
                res_text = ''
                if response.status_code == 200:
                    res_text = ','.join(word['word'] for word in response.json()['data'])
                    print("output: ", res_text)
                    break
                            
                else:
                    print(f"请求失败，状态码:{response.status_code}")
            except Exception as e:
                print(e)

        return (res_text,)