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

class WORDCARD:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        
        self.keys = json.load(open(os.path.join(os.path.dirname(__file__).split('/api/')[0], "keys.json"), "r"))
        self.headers = {
            "ailab-web-session": self.keys["api"]["ailab-web-session"],
             "Content-Type": "application/json"
        }
        self.url = self.keys["wordcard"]["url"]
        self.payload = {
            "appId": self.keys["api"]["appId"],
            "clientId": self.keys["api"]["clientId"],
        }
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING",),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/api/aigc"
    def test(self, prompt, random_seed=0):
        np.random.seed(random_seed)
        print("开始调用接口：wordcard")
        print("prompt: ", prompt)
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        # Send the image to the server
        for i in range(3):
            print(f"第{i+1}次请求")
            try:
                data = {
                    "appId": self.keys["api"]["appId"],
                    "clientId": self.keys["api"]["clientId"],
                    "text": prompt
                }   
                response = requests.post(self.url, headers=self.headers, json=data, timeout=60)
                # 打印响应结果
                if response.status_code == 200:
                    print("请求成功:", response.text)
                    data_dict = json.loads(response.text)
                    img_url = data_dict['generated_text']
                    img_response = requests.get(img_url)
                    if img_response.status_code == 200:
                        with open(tmp_img_path, 'wb') as f:
                            f.write(img_response.content)
                        print("图片下载成功")
                        break
                    else:
                        print("图片下载失败:", img_response.status_code)
                else:
                    print("请求失败:", response.status_code, response.text)
            except Exception as e:
                print(e)

        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        # return (image,)
        return (img,)