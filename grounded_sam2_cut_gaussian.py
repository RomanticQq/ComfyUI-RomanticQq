# 保持裁剪后的图片比例与原图一致
import os
import cv2
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "Grounded-SAM-2"))
import torch
import uuid
import torch
import requests
import numpy as np
from torchvision.ops import box_convert
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
from torchvision.ops import box_convert
# sys.path.insert(0, '/root/project/ComfyUI/custom_nodes/ComfyUI-RomanticQq/Grounded-SAM-2')
sys.path.insert(0, f'{os.path.dirname(__file__)}/Grounded-SAM-2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict

class GroundedSam2CutGaussian:
    def __init__(self):
        self.dir_name = os.path.dirname(__file__)
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        """
        1. 配置 Hyper parameters
        """
        # 模型地址
        SAM2_CHECKPOINT = f"{self.dir_name}/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        SAM2_MODEL_CONFIG = f"configs/sam2.1/sam2.1_hiera_l.yaml"
        GROUNDING_DINO_CONFIG = f"{self.dir_name}/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT = f"{self.dir_name}/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_prompt = "subject"
        self.box_threshold = 0.35
        self.text_threshold = 0.25

        """
        2. 加载模型
        """
        # build SAM2 image predictor
        sam2_checkpoint = SAM2_CHECKPOINT
        model_cfg = SAM2_MODEL_CONFIG
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # load grounding dino model
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=self.DEVICE
        )

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
            self.text_prompt = text_prompt
            self.box_threshold = box_threshold
            self.text_threshold = text_threshold
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
            cropped_object, cropped_blur_object = self.process_single_image(tmp_img_path)
            if cropped_blur_object is not None:
                cv2.imwrite(tmp_img_path, cropped_blur_object)
                print(f"Finished processing {tmp_img_path}")
        except Exception as e:
            print(f"Error processing image: {e}")
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        # return (image,)
        return (img,)
    def process_single_image(self, img_path):
        '''
        处理单张图片，输出裁剪后的保持输入比例的目标区域
        '''
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            image_source, image = load_image(img_path)
            self.sam2_predictor.set_image(image_source)
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image,
                caption=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            img = cv2.imread(img_path)
            if len(input_boxes) > 0 and masks.shape[0] > 0:
                h, w, _ = image_source.shape
                original_aspect_ratio = w / h
                x1, y1, x2, y2 = input_boxes[0].astype(int)
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w <= 0 or box_h <= 0:
                    return None, None
                box_aspect_ratio = box_w / box_h
                if box_aspect_ratio > original_aspect_ratio:
                    # Box is wider than original image ratio -> increase height
                    target_w = box_w
                    target_h = int(target_w / original_aspect_ratio)
                else:
                    # Box is taller than original image ratio (or equal) -> increase width
                    target_h = box_h
                    target_w = int(target_h * original_aspect_ratio)

                # Calculate padding needed
                pad_w = target_w - box_w
                pad_h = target_h - box_h

                # Calculate initial new coordinates (distribute padding)
                x1_new = x1 - pad_w // 2
                y1_new = y1 - pad_h // 2
                # Use target dimensions to avoid compounding rounding errors initially
                x2_new = x1_new + target_w 
                y2_new = y1_new + target_h

                # Adjust coordinates to stay within image bounds [0, w) and [0, h)
                # Calculate necessary shifts if boundaries are exceeded
                shift_x = 0
                if x1_new < 0:
                    shift_x = -x1_new # Need to shift right by this amount
                elif x2_new > w:
                    shift_x = w - x2_new # Need to shift left by this amount (negative value)

                shift_y = 0
                if y1_new < 0:
                    shift_y = -y1_new # Need to shift down by this amount
                elif y2_new > h:
                    shift_y = h - y2_new # Need to shift up by this amount (negative value)

                # Apply shifts
                x1_new += shift_x
                x2_new += shift_x
                y1_new += shift_y
                y2_new += shift_y

                # Ensure coordinates are still within bounds after shift (can happen in edge cases)
                x1_new = max(0, x1_new)
                y1_new = max(0, y1_new)
                x2_new = min(w, x2_new)
                y2_new = min(h, y2_new)

                # Recalculate dimensions after clamping/shifting
                current_crop_w = x2_new - x1_new
                current_crop_h = y2_new - y1_new

                # Final aspect ratio adjustment: Check if one dimension needs adjustment based on the other
                # Due to clamping, the aspect ratio might be off. Try to adjust the *smaller* dimension outwards if possible.
                target_h_based_on_w = int(current_crop_w / original_aspect_ratio)
                target_w_based_on_h = int(current_crop_h * original_aspect_ratio)

                # Try expanding height if it's smaller than target and fits
                if current_crop_h < target_h_based_on_w:
                    diff_h = target_h_based_on_w - current_crop_h
                    pad_top = diff_h // 2
                    pad_bottom = diff_h - pad_top
                    # Center expansion if possible
                    if y1_new >= pad_top and y2_new + pad_bottom <= h:
                        y1_new -= pad_top
                        y2_new += pad_bottom
                    # Else, expand downwards if possible
                    elif y2_new + diff_h <= h:
                        y2_new += diff_h
                    # Else, expand upwards if possible
                    elif y1_new >= diff_h:
                        y1_new -= diff_h
                    # Update height after potential expansion
                    current_crop_h = y2_new - y1_new

                # Try expanding width if it's smaller than target and fits (use updated height)
                # Recalculate target width based on potentially adjusted height
                target_w_based_on_h_final = int(current_crop_h * original_aspect_ratio)
                if current_crop_w < target_w_based_on_h_final:
                    diff_w = target_w_based_on_h_final - current_crop_w
                    pad_left = diff_w // 2
                    pad_right = diff_w - pad_left
                    if x1_new >= pad_left and x2_new + pad_right <= w:
                        x1_new -= pad_left
                        x2_new += pad_right
                    elif x2_new + diff_w <= w:
                        x2_new += diff_w
                    elif x1_new >= diff_w:
                        x1_new -= diff_w
                x1_final, y1_final, x2_final, y2_final = map(int, [x1_new, y1_new, x2_new, y2_new])
                if x1_final < x2_final and y1_final < y2_final:
                    # 这一步仅裁剪目标区域，不对背景模糊处理
                    cropped_object = img.copy()[y1_final:y2_final, x1_final:x2_final]

                    # --- Masking and Cropping Masked Image ---
                    first_mask = masks[0].astype(np.uint8)
                    if first_mask.ndim == 3:
                        first_mask = np.squeeze(first_mask, axis=-1)

                    # Apply Gaussian blur to the background
                    blurred_background = cv2.GaussianBlur(img, (25, 25), 0)
                    masked_img_blurred_bg = blurred_background.copy()
                    boolean_mask = first_mask.astype(bool)

                    # Ensure boolean_mask is broadcastable to img shape for boolean indexing
                    # This usually works if img is (H, W, C) and boolean_mask is (H, W)
                    try:
                        masked_img_blurred_bg[boolean_mask] = img[boolean_mask] # Copy original object pixels over blurred bg
                    except IndexError:
                        print(f"Warning: Mask shape {boolean_mask.shape} mismatch with image shape {img.shape} for {base_name}. Skipping masked crop.")
                        return None, None

                    masked_img = masked_img_blurred_bg

                    # Crop the masked image using the same calculated coordinates
                    cropped_blur_object = masked_img[y1_final:y2_final, x1_final:x2_final]
                    return cropped_object, cropped_blur_object
                else:
                    return None, None
            else:
                return None, None
        except Exception:
            return None, None
