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

class JIMENG_T2I_3:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.url = 'http://test-api.aiedevice.com/interact/vui/ai/v1'  # 替换为实际接口URL
        self.keys = json.load(open(os.path.join(os.path.dirname(__file__).split('/api/')[0], "keys.json"), "r"))
        self.headers = {
            "RC-DEVICE-SESSION": self.keys["api"]["RC-DEVICE-SESSION"],
            "ailab-web-session": self.keys["api"]["ailab-web-session"],
            "server-super-token": self.keys["api"]["server-super-token"]
        }
    @classmethod
    def INPUT_TYPES(s):
        modle_names = ['jimeng-3.0', 'jimeng-3.1', 'jimeng-4.0']
        return {
            "required": {
                "prompt": ("STRING",),
                "width": ("INT", {"default": 1024, "min": 0, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 1536, "min": 0, "max": 2048, "step": 1}),
                "model_name": (modle_names,),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1, "control_after_generate": True}),
                "imgurl": ("STRING",{"default": None} ),

            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/api/jimeng"
    def test(self, prompt, width, height, model_name, random_seed, imgurl=None):
        np.random.seed(random_seed)
        print("prompt: ", prompt)
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        # Send the image to the server
        for i in range(3):
            print(f"第{i+1}次请求")
            try:
                if model_name == 'jimeng-3.0':
                    data = self.jimeng_3(prompt, width, height)
                    print("开始调用接口：jimeng_t2i_3.0")
                elif model_name == 'jimeng-3.1':
                    data = self.jimeng_3_1(prompt, width, height)
                    print("开始调用接口：jimeng_t2i_3.1")
                elif model_name == 'jimeng-4.0':
                    data = self.jimeng_4(prompt, width, height, imgurl)
                    print("开始调用接口：jimeng_t2i_4.0")
                json_data = json.dumps(data)
                response = requests.post(self.url, headers=self.headers, data=json_data)
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
    def jimeng_3(self, prompt, width, height):
        data = {
            "appId": self.keys["api"]["appId"],
            "clientId": self.keys["api"]["clientId"],
            "token": self.keys["api"]["token"],
            "type": 3,
            "model": 60,
            "text": prompt,
            "parameters": {
                "width": width,
                "height": height
            },
        }
        return data

    def jimeng_3_1(self, prompt, width, height):
        data = {
            "appId": self.keys["api"]["appId"],
            "clientId": self.keys["api"]["clientId"],
            "token": self.keys["api"]["token"],
            "type": 3,
            "model": 60,
            "text": prompt,
            "parameters": {
                "width": width,
                "height": height,
                "version": "3.1",
            },
        }
        return data
    
    def jimeng_4(self, prompt, width, height, imgurl=None):
        data = {
            "appId": self.keys["api"]["appId"],
            "clientId": self.keys["api"]["clientId"],
            "token": self.keys["api"]["token"],
            "type": 3,
            "model": 95,
            "inputs": [
                {
                    "type": "text",
                    "content": prompt
                }
            ],
            "parameters": {
                "width": width,
                "height": height,
            }
        }
        if imgurl:
            data["inputs"].append({
                "type": "image",
                "content": imgurl
            })
        return data