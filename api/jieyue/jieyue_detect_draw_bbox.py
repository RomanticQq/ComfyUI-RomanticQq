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

class JIeyueDetectDrawBboxAPI:
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

    RETURN_TYPES = ("STRING","IMAGE",)
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
                tmp_img_path = os.path.join(self.tmp_dir, f"{uuid.uuid4()}.jpg")
                img_response = requests.get(imageUrl)
                if img_response.status_code == 200:
                    with open(tmp_img_path, 'wb') as f:
                        f.write(img_response.content)
                    print(f"Image downloaded from {imageUrl} and saved to {tmp_img_path}")
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
                    # -------------------------------------------------

                    img = cv2.imread(tmp_img_path)
                    for kv in response.json()['data']:        
                        # 获取位置信息
                        leftX = kv['position']['leftX']
                        rightX = kv['position']['rightX']
                        leftY = kv['position']['leftY']
                        rightY = kv['position']['rightY']
                        
                        # 计算矩形框的左上角和右下角坐标
                        # 注意：需要根据实际的坐标系统调整
                        pt1 = (int(leftX), int(leftY))  # 左上角 (x, y)
                        pt2 = (int(rightX), int(rightY))  # 右下角 (x, y)
                        
                        # 生成随机颜色 (BGR格式)
                        # 确保颜色足够明显：至少有一个通道值较高
                        color = (
                            np.random.randint(50, 256),  # B通道
                            np.random.randint(50, 256),  # G通道
                            np.random.randint(50, 256)   # R通道
                        )
                        
                        # 在图像上绘制矩形框
                        # cv2.rectangle(img, pt1, pt2, color, thickness)
                        # color: BGR格式，随机生成的颜色
                        # thickness: 线条粗细，-1表示填充，正数表示线条宽度
                        cv2.rectangle(img, pt1, pt2, color, 2)  # 随机颜色矩形框，线宽2
                        
                        # 可选：在矩形框上方添加文字标签（使用相同的颜色）
                        word = kv['word']
                        cv2.putText(img, word, (int(leftX)+10, int(leftY) + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    
                    # 保存绘制了矩形框的图像
                    cv2.imwrite(tmp_img_path, img)
                    print(f"Annotated image saved to {tmp_img_path}")
                    break       
                else:
                    print(f"请求失败，状态码:{response.status_code}")
            except Exception as e:
                print(e)
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        return (res_text,img,)