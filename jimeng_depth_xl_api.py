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

class Jimeng_Depth_Xl:
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
                "imageUrl": ("STRING",),
                "strength": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    def test(self, prompt, imageUrl, strength):
        print("开始调用接口：即梦景深XL")
        print("prompt: ", prompt)
        print("imageUrl: ", imageUrl)
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        # Send the image to the server
        for i in range(3):
            print(f"第{i+1}次请求")
            try:
                data = {
                    "appId": "007cd3983760",
                    "clientId": "F6010A0A000005",
                    "token": "c1970c0208d7430ebddae7041afc90d9",
                    "type": 3,
                    "model": 61,
                    "imageUrl": imageUrl,
                    "text": prompt,
                    "parameters": {
                            "controlnet_args": [
                                {
                                    "type": "depth",
                                    "strength": strength,
                                    "binary_data_index": 0
                                }
                            ]
                        }
                }   
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