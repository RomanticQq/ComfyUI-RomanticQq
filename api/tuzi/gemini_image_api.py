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

class GeminiImageAPI:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.keys = json.load(open(os.path.join(os.path.dirname(__file__).split('/api')[0], "keys.json"), "r"))
        self.client = OpenAI(
            base_url=self.keys["tuzi"]["default"]["api_base"],
            api_key=self.keys["tuzi"]["default"]["api_key"]
        )
    @classmethod
    def INPUT_TYPES(s):
        modle_names = ["gemini-2.5-flash-image-vip", "gemini-2.5-flash-image"]
        return {
            "required": {
                "prompt": ("STRING",),
                "model_name": (modle_names,),
                "random_seed": ("INT", {"default": 66666, "min": 0, "max": 2**32 - 1, "step": 1, "control_after_generate": True}),
            },
            "optional": {
                "imageUrl": ("ListString",{"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/api/tuzi"
    def test(self, prompt, model_name,imageUrl=None, random_seed=0):
        np.random.seed(random_seed)
        print(os.getcwd())
        print("prompt: ", prompt)
        # Send the image to the server
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        for i in range(3):
            print(f"第{i+1}次请求")
            try:
                if imageUrl==None or imageUrl=="":
                    result = self.client.images.generate(
                        model=model_name,
                        prompt=prompt
                    )
                else:
                    img_list = []
                    for idx, url in enumerate(imageUrl):
                        tmp_edit_img_path = os.path.join(self.tmp_dir, f"{uuid.uuid4()}.jpg")
                        response = requests.get(url)
                        response.raise_for_status()
                        with open(tmp_edit_img_path, "wb") as f:
                            f.write(response.content)
                        img_list.append(open(tmp_edit_img_path, "rb"))
                        os.remove(tmp_edit_img_path)
                    result = self.client.images.edit(
                        model=model_name,
                        image=img_list,
                        prompt=prompt,
                    )
                image_base64 = result.data[0].b64_json
                image_url = result.data[0].url

                if image_base64:
                    image_bytes = base64.b64decode(image_base64)
                    with open(tmp_img_path, "wb") as f:
                        f.write(image_bytes)
                    print(f"图片已通过base64保存为 {tmp_img_path}")
                elif image_url:
                    response = requests.get(image_url)
                    response.raise_for_status()
                    with open(tmp_img_path, "wb") as f:
                        f.write(response.content)
                    print(f"图片已通过url下载并保存为 {tmp_img_path}")
                else:
                    raise ValueError("API 没有返回图片的 base64 数据或图片链接！")
                break

            except Exception as e:
                print(e)
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        return (img,)