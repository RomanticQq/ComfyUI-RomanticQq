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

class CustomAPI:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.url = 'http://test-api.aiedevice.com/interact/vui/ai/v1'  # 替换为实际接口URL
        self.keys = json.load(open(os.path.join(os.path.dirname(__file__).split('/api')[0], "keys.json"), "r"))
        self.headers = {
            "RC-DEVICE-SESSION": self.keys["api"]["RC-DEVICE-SESSION"],
            "ailab-web-session": self.keys["api"]["ailab-web-session"],
            "server-super-token": self.keys["api"]["server-super-token"]
        }
    @classmethod
    def INPUT_TYPES(s):
        modle_names = ['Doubao', 'qwen-vl', 'qwen-max', 'qwen-turbo', 'qwen-long', 'qwen-plus', 'qwen-omni', 'qwen-flash']
        return {
            "required": {
                "prompt": ("STRING",),
                "model_name": (modle_names,),
                "imageUrl": ("STRING",{"default": None}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1, "control_after_generate": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/api"
    def test(self, prompt, model_name,imageUrl=None, random_seed=0):
        np.random.seed(random_seed)
        # model_id 对应模型
        model_kv = {
            'Doubao':22,
            'qwen-vl': 26,
            'qwen-max': 27,
            'qwen-turbo': 28,
            'qwen-long':29,
            'qwen-plus':45,
            'qwen-omni':63,
            'qwen-flash':81,
        }

        print(os.getcwd())
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
                    "model": model_kv[model_name],
                    "stream": 1,
                    "text": prompt,
                }

                if imageUrl is not None:
                    data["imageUrl"] = imageUrl

                json_data = json.dumps(data)

                response = requests.post(self.url, headers=self.headers, data=json_data, stream=True, timeout=5)

                res_text = ''
                if response.status_code == 200:
                    for chunk in response.iter_lines():
                        if chunk:
                            str_arr = chunk.decode('utf-8').split("data:")[1:]
                            res_json = json.loads(str_arr[0])
                            if res_json['is_finished'] == True:
                                break
                            text = res_json['generated_text']
                            res_text = res_text + text

                    print("input: ", prompt)
                    print("output: ", res_text)
                    break
                            
                else:
                    print(f"请求失败，状态码:{response.status_code}")
            except Exception as e:
                print(e)

        return (res_text,)