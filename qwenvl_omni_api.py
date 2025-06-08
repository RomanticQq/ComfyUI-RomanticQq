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

class QWENVL_Omni:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.url = 'http://test-api.aiedevice.com/interact/vui/ai/v1'  # 替换为实际接口URL
        self.keys = json.load(open(os.path.join(os.path.dirname(__file__), "keys.json"), "r"))
        self.headers = {
            "RC-DEVICE-SESSION": self.keys["api"]["RC-DEVICE-SESSION"],
            "ailab-web-session": self.keys["api"]["ailab-web-session"],
            "server-super-token": self.keys["api"]["server-super-token"]
        }
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING",),
                "imageUrl": ("STRING",{"default": None}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    def test(self, prompt, imageUrl=None):
        print("prompt: ", prompt)
        # Send the image to the server
        for i in range(3):
            print(f"第{i+1}次请求")
            try:
                data = {
                    "appId": self.keys["api"]["appId"],
                    "clientId": self.keys["api"]["clientId"],
                    "token": self.keys["api"]["token"],
                    "type": 3,
                    "model": 63,
                    "stream": 1,
                    "text": prompt
                }

                if imageUrl is not None:
                    data["imageUrl"] = imageUrl

                json_data = json.dumps(data)

                response = requests.post(self.url, headers=self.headers, data=json_data, stream=True, timeout=5)

                res_text = ''
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            a = chunk.decode('utf-8')
                            str_arr = chunk.decode('utf-8').split("data:")[1:]
                            text = "".join([json.loads(s)['generated_text'] for s in str_arr])
                            res_text = res_text + text

                    print("input: ", prompt)
                    print("output: ", res_text)
                    break
                            
                else:
                    print(f"请求失败，状态码：{response.status_code}")
            except Exception as e:
                print(e)

        return (res_text,)