import os
import cv2
import uuid
import torch
import numpy as np
import json
import requests
from minio import Minio
from datetime import datetime, timedelta
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

class FluxGymCaption:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        print("tmp_dir:", self.tmp_dir)
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device={self.device}")
        self.torch_dtype = torch.float16
        print("开始加载模型")
        self.model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_TYPES_NAMES = ("caption_text",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    def test(self, image):
        tmp_img_name = str(uuid.uuid4()) + ".png"
        tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
        img = image.numpy()[0]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_img_path, img)
        print(f"tmp_img_path: {tmp_img_path}")
        image_path = tmp_img_path  # Replace with your image path or PIL Image object
        if isinstance(image_path, str):  # If image is a file path
            image = Image.open(image_path).convert("RGB")
        prompt = "<DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        print(f"inputs {inputs}")

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        print(f"generated_ids {generated_ids}")

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"generated_text: {generated_text}")
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        print(f"parsed_answer = {parsed_answer}")
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        print(f"caption_text = {caption_text}")
        os.remove(tmp_img_path)
        return (caption_text,)
