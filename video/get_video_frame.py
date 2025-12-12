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
from PIL import Image, ImageDraw, ImageOps

class GetFirstLastVideoFrames:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/video"
    def test(self, video):
        # video type: <class 'comfy_api.latest._input_impl.video_types.VideoFromFile'>
        # video dir: ['_VideoFromFile__file', '__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__',
        #              '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__',
        #                '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', 
        #              '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', 
        #              '_abc_impl', '_get_first_video_stream', 'get_components', 'get_components_internal', 'get_container_format', 
        #              'get_dimensions', 'get_duration', 'get_frame_count', 'get_frame_rate', 'get_stream_source', 'save_to']
        tmp_video_name = str(uuid.uuid4()) + ".mp4"
        tmp_video_path = os.path.join(self.tmp_dir, tmp_video_name)
        video.save_to(tmp_video_path)
        cap = cv2.VideoCapture(tmp_video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {tmp_video_path}")

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("无法读取首帧")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise ValueError("视频帧数异常")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("无法读取尾帧")

        first_path = os.path.join(self.tmp_dir, f"{uuid.uuid4()}_first.png")
        last_path = os.path.join(self.tmp_dir, f"{uuid.uuid4()}_last.png")

        cv2.imwrite(first_path, first_frame)
        cv2.imwrite(last_path, last_frame)


        first_img = cv2.imread(first_path)
        first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
        first_img = torch.from_numpy(np.expand_dims(first_img, axis=0) / 255.0)

        last_img = cv2.imread(last_path)
        last_img = cv2.cvtColor(last_img, cv2.COLOR_BGR2RGB)
        last_img = torch.from_numpy(np.expand_dims(last_img, axis=0) / 255.0)

        return (first_img, last_img)