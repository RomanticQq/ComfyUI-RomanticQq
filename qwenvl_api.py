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

class QWENVL:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.url = 'http://test-api.aiedevice.com/interact/vui/ai/v1'  # 替换为实际接口URL
        self.headers = {
            "RC-DEVICE-SESSION": "c1970c0208d7430ebddae7041afc90d9",
            "ailab-web-session": "12312412",
            "server-super-token": "e2fb7af59517ac82d640c60df5365e61"
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
                    "appId": "007cd3983760",
                    "clientId": "F6010A0A000005",
                    "token": "c1970c0208d7430ebddae7041afc90d9",
                    "type": 3,
                    "model": 26,
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