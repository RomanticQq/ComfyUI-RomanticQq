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

class MINIO_UPLOAD:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        self.minio_dir = "upload"
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.keys = json.load(open(os.path.join(os.path.dirname(__file__).split('/api/')[0], "keys.json"), "r"))
        endpoint = self.keys["minio"]["endpoint"]
        access_key = self.keys["minio"]["access_key"]
        secret_key = self.keys["minio"]["secret_key"]
        self.bucket_name = 'myminio'
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_TYPES_NAMES = ("minio_url",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/upload"
    def test(self, image):
        tmp_img_name = str(uuid.uuid4()) + ".jpg"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        minio_img_path = os.path.join(self.minio_dir, tmp_img_name)
        img = image.numpy()[0]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_img_path, img)
        # Upload the image to MinIO
        with open(tmp_img_path, 'rb') as f:
            self.client.put_object(self.bucket_name, minio_img_path, f, os.path.getsize(tmp_img_path), content_type='image/jpeg')
        minio_url = self.client.get_presigned_url("GET", self.bucket_name, minio_img_path, expires=timedelta(days=1)).split("?")[0]
        os.remove(tmp_img_path)
        return (minio_url,)
