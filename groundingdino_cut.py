import os
import cv2
import uuid
import json
import torch
import requests
import numpy as np
from torchvision.ops import box_convert
from datetime import datetime, timedelta
from groundingdino.util.inference import load_model, load_image, predict, annotate

class GroundingDino:
    def __init__(self):
        self.dir_name = os.path.dirname(__file__)
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model = load_model(f"{self.dir_name}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", f"{self.dir_name}/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "image": ("IMAGE", {"default": None, "forceInput": True}),
                "text_prompt": ("STRING", {"default": "subject"}),
                "box_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "imageUrl": ("STRING", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_TYPES_NAMES = ("image",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    def test(self, text_prompt="subject", box_threshold=0.35, text_threshold=0.25, image=None, imageUrl=None):
        try:
            tmp_img_name = str(uuid.uuid4()) + ".jpg"
            tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
            if image is not None:
                img = image.numpy()[0]
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(tmp_img_path, img)
            elif imageUrl is not None:
                response = requests.get(imageUrl)
                if response.status_code == 200:
                    with open(tmp_img_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Image downloaded from {imageUrl} and saved to {tmp_img_path}")
                else:
                    raise ValueError(f"Failed to download image from {imageUrl}, status code: {response.status_code}")
            else:
                raise ValueError("Either 'image' or 'imageUrl' must be provided.")
            image_source, image = load_image(tmp_img_path)
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            if len(boxes) == 0:
                raise(f"No objects found in image {tmp_img_name}")
            max_confidence_index = torch.argmax(logits)
            boxes = boxes[max_confidence_index:max_confidence_index + 1]
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            x1, y1, x2, y2 = [int(num) for num in input_boxes[0]]
            origin_image = cv2.imread(tmp_img_path)
            cut_image = origin_image[y1:y2, x1:x2]
            cv2.imwrite(tmp_img_path, cut_image)
            print(f"Saved cut image to {tmp_img_path}")
        except Exception as e:
            print(f"Error processing image: {e}")
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        # return (image,)
        return (img,)
