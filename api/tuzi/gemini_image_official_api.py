import os
import cv2
import uuid
import torch
import numpy as np
import base64
import os
import json
import requests
from minio import Minio
from datetime import datetime, timedelta
from openai import OpenAI
import http.client
import json
import base64


class GeminiImageOfficialAPI:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.keys = json.load(open(os.path.join(os.path.dirname(__file__).split('/api')[0], "keys.json"), "r"))
        if "tuzi" in self.keys.keys() and "gemini_official" in self.keys["tuzi"].keys():
            self.headers = {
            'Authorization': f"Bearer {self.keys["tuzi"]["gemini_official"]["api_key"]}",
            'Content-Type': 'application/json'
            }
    @classmethod
    def INPUT_TYPES(s):
        aspectRatio_arr= ['1:1','2:3','3:2','3:4','4:3','4:5','5:4','9:16','16:9','21:9']
        return {
            "required": {
                "prompt": ("STRING",),
                "aspectRatio": (aspectRatio_arr, {"default": '1:1'}),
                "random_seed": ("INT", {"default": 66666, "min": 0, "max": 2**32 - 1, "step": 1, "control_after_generate": True}),
            },
            "optional": {
                "imageUrl": ("STRING",{"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/api/tuzi"
    def test(self, prompt,aspectRatio, imageUrl=None, random_seed=0):
        if "tuzi" not in self.keys.keys() or "gemini_official" not in self.keys["tuzi"].keys():
            raise ValueError("Tuzi Gemini Official API keys are not configured properly.")
        conn = http.client.HTTPSConnection("api.tu-zi.com")
        np.random.seed(random_seed)
        print(os.getcwd())
        print("prompt: ", prompt)
        # Send the image to the server
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        for i in range(3):
            print(f"第{i+1}次请求")
            try:
                config = {
                    "contents": [
                        {
                            "parts": [
                                {
                                "text": prompt
                                },
                            ]
                        }
                    ],
                    "generationConfig": {
                        "responseModalities": [
                            "IMAGE",
                            # "TEXT"
                        ],
                        "imageConfig": {
                            "aspectRatio": aspectRatio
                        }
                    }
                    }
                print("--------------------------")
                print(imageUrl)
                print("--------------------------")
                if imageUrl:
                    tmp_edit_img_path = os.path.join(self.tmp_dir, f"{uuid.uuid4()}.jpg")
                    response = requests.get(imageUrl)
                    response.raise_for_status()
                    with open(tmp_edit_img_path, "wb") as f:
                        f.write(response.content)
                    input_image_base64=base64.b64encode(open(tmp_edit_img_path, 'rb').read()).decode('utf-8')
                
                    config['contents'][0]['parts'].append({"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": input_image_base64
                    }})
                # print("--------------------------")
                # print(config)
                # print("--------------------------")
                payload = json.dumps(config)
                conn.request("POST", "/v1beta/models/gemini-2.5-flash-image:generateContent", payload, self.headers)
                res = conn.getresponse()
                data = res.read()
                data = data.decode("utf-8")
                # print(data)
                # base64_img=json.loads(data)["candidates"][0]['content']['parts'][0]['inlineData']['data']
                parts=json.loads(data)["candidates"][0]['content']['parts']
                if 'inlineData' in parts[0].keys():
                    base64_img=parts[0]['inlineData']['data']
                else:
                    base64_img=parts[1]['inlineData']['data']
                img=base64.b64decode(base64_img)
                with open(tmp_img_path, 'wb') as f:
                    f.write(img)
                os.remove(tmp_edit_img_path)
                break
            except Exception as e:
                print(e)
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        return (img,)