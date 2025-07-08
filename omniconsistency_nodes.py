import os
import torch
import numpy as np
from PIL import Image

from .omniconsistency_src_inference.pipeline import FluxPipeline
from .omniconsistency_src_inference.lora_helper import set_single_lora, unset_lora

# -----------------------------------------------------------------------------
#  helper to clear cached key/value banks --------------------------------------
# -----------------------------------------------------------------------------

def clear_cache(transformer):
    """Clear KV cache created by OmniConsistency attention processors."""
    for _, attn_proc in transformer.attn_processors.items():
        if hasattr(attn_proc, "bank_kv"):
            attn_proc.bank_kv.clear()

# -----------------------------------------------------------------------------
#  Singleton – heavy objects are loaded once per session                        
# -----------------------------------------------------------------------------

class _OmniGeneratorSingleton:
    _pipe: FluxPipeline | None = None
    _initialized: bool = False

    _last_base: str | None = None
    _last_omni: str | None = None
    _last_lora: str | None = None
    _last_lora_scale: float | None = None

    @classmethod
    def _rebuild_pipeline(cls, base_model_path: str, dtype: torch.dtype, device: str):
        print("[OmniConsistency] Loading FLUX base model …")
        pipe = FluxPipeline.from_pretrained(base_model_path, torch_dtype=dtype).to(device)
        cls._pipe = pipe
        cls._initialized = True

    @classmethod
    def initialize(
        cls,
        base_model_path: str,
        omni_model_path: str,
        lora_path: str | None,
        lora_scale: float = 1.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> FluxPipeline:
        """Ensure pipeline is ready and matches the requested paths."""

        # 1. First‑time load or base model changed → full rebuild
        if (not cls._initialized) or (base_model_path != cls._last_base):
            if cls._initialized:
                # try to free old pipe to save VRAM
                try:
                    cls._pipe.to("cpu")  # type: ignore
                except Exception:
                    pass
                del cls._pipe
                torch.cuda.empty_cache()
            cls._rebuild_pipeline(base_model_path, dtype, device)

        pipe = cls._pipe  # type: ignore

        # 2. OmniConsistency LoRA changed
        if omni_model_path != cls._last_omni:
            print("[OmniConsistency] Switching OmniConsistency model …")
            unset_lora(pipe.transformer)
            set_single_lora(pipe.transformer, omni_model_path, lora_weights=[1], cond_size=512)

        # 3. Extra LoRA changed
        if lora_path != cls._last_lora or lora_scale != cls._last_lora_scale:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
            if lora_path:
                print("[OmniConsistency] Loading user LoRA …")
                folder, name = os.path.split(lora_path)
                if folder == "":
                    folder = "."
                if round(lora_scale, 2) == 1.00:
                    print("[OmniConsistency] Using LoRA scale 1.00, loading without adapter name …")
                    pipe.load_lora_weights(folder, weight_name=name)
                else:
                    print(f"[OmniConsistency] Using LoRA scale {lora_scale:.2f}, loading with adapter name 'style' …")
                    pipe.load_lora_weights(folder, weight_name=name, adapter_name="style")
                    pipe.set_adapters(["style"], adapter_weights=[lora_scale])
                # pipe.set_adapters("style1", adapter_weights=1.0)

        # 4. Record current state
        cls._last_base = base_model_path
        cls._last_omni = omni_model_path
        cls._last_lora = lora_path
        cls._last_lora_scale = lora_scale
        return pipe

    @classmethod
    def generate(
        cls,
        prompt: str,
        spatial_pil: Image.Image,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        progress_callback=None,
    ) -> Image.Image:
        if not cls._initialized:
            raise RuntimeError("Pipeline not initialised; call initialize() first!")

        generator_cpu = torch.Generator(device="cpu").manual_seed(seed)

        image = cls._pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            spatial_images=[spatial_pil],
            subject_images=[],
            cond_size=512,
            generator=generator_cpu,
            callback_on_step_end=progress_callback,
            callback_on_step_end_tensor_inputs=[],
        ).images[0]
        return image

# -----------------------------------------------------------------------------
#  ComfyUI custom node                                                          
# -----------------------------------------------------------------------------

class Comfyui_OmniConsistency:
    CATEGORY = "RomanticQq"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

    @staticmethod
    def INPUT_TYPES():
        wide = {"uiWidth": 400, "multiline": False}
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "3D Chibi style, Three individuals standing together in the office."}),
                "spatial_image": ("IMAGE",),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "width":  ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "seed":   ("INT", {"default": 42, "min": 0, "max": 2**32-1}),
                "base_model_path": ("STRING", wide | {"default": "black-forest-labs/FLUX.1-dev"}),
                "omni_model_path": ("STRING", wide | {"default": "/path/to/OmniConsistency.safetensors"}),
                "lora_scale": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
            },
            "optional": {
                "lora_path": (
                    "STRING",
                    wide | {
                        "default": "",
                        "placeholder": "extra LoRA .safetensors (leave blank = none)",
                        "tooltip": "feel free to load any flux.1-based LoRA"
                    },
                ),
            }
        }

    # ---------- image conversion helper -----------------------------------
    # ---------------- util: convert to PIL ----------------------------------
    @staticmethod
    def _ndarray_to_pil(arr: np.ndarray) -> Image.Image:
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(arr)

    # -----------------------------------------------------------------------
    def convert_image_to_pil(self, img):
        """
        将 ComfyUI IMAGE 端口可能出现的任何格式转换为 PIL.Image：
          • list / tuple
          • dict  (常见键: "image", "samples")
          • np.ndarray  (HWC / NHWC / 4D batch)
          • torch.Tensor (C,H,W | H,W,C | N,C,H,W | N,H,W,C | H,W)
          • PIL.Image
        """
        # 1) list / tuple ── 取第一张
        if isinstance(img, (list, tuple)):
            if not img:
                raise TypeError("Empty image list passed to node.")
            return self.convert_image_to_pil(img[0])

        # 2) dict ── 常见包装
        if isinstance(img, dict):
            for key in ("image", "samples", "img"):
                if key in img:
                    return self.convert_image_to_pil(img[key])

        # 3) 已经是 PIL
        if isinstance(img, Image.Image):
            return img

        # 4) NumPy
        if isinstance(img, np.ndarray):
            if img.ndim == 4:                       # N,H,W,C
                img = img[0]
            if img.ndim == 3:                       # H,W,C
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                return Image.fromarray(img)
            if img.ndim == 2:                       # H,W  → 灰度
                img = np.stack([img]*3, axis=-1)
                return self.convert_image_to_pil(img)
            raise TypeError(f"Unsupported ndarray shape {img.shape}")

        # 5) torch.Tensor
        if torch.is_tensor(img):
            img = img.detach().cpu()
            if img.ndim == 4:                       # N,C,H,W  or N,H,W,C
                img = img[0]
            if img.ndim == 3:
                # C,H,W → permute；H,W,C → 原样
                if img.shape[0] in (1, 3):          # C,H,W
                    img = img.permute(1, 2, 0)
                elif img.shape[-1] not in (1, 3):   # H,W,C 检查
                    raise TypeError(f"Ambiguous tensor shape {img.shape}")
                img = img.numpy()
                return self.convert_image_to_pil(img)
            if img.ndim == 2:                       # H,W
                img = img.unsqueeze(0)              # 1,H,W
                return self.convert_image_to_pil(img)
            raise TypeError(f"Unsupported tensor shape {img.shape}")

        # 6) 其他
        raise TypeError(f"Unsupported image type: {type(img)}")

    # ---------------- main -------------------------------------------------
    def generate(
        self,
        prompt,
        spatial_image,
        height,
        width,
        guidance_scale,
        num_inference_steps,
        seed,
        base_model_path,
        omni_model_path,
        lora_path="",
        lora_scale=1.0,
    ):
        
        print("[OmniConsistency] Initializing OmniConsistency pipeline …")
        _OmniGeneratorSingleton.initialize(
            base_model_path=base_model_path,
            omni_model_path=omni_model_path,
            lora_path=lora_path if lora_path else None,
            lora_scale=lora_scale,
        )

        spatial_pil = self.convert_image_to_pil(spatial_image)

        # progress bar ------------------------------------------------------
        from comfy.utils import ProgressBar
        pbar = ProgressBar(num_inference_steps)
        def _cb(*_):
            pbar.update(1)
            return {}

        img_pil = _OmniGeneratorSingleton.generate(
            prompt=prompt,
            spatial_pil=spatial_pil,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            progress_callback=_cb,
        )

        # HWC → tensor
        img_np = np.asarray(img_pil, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).contiguous()

        clear_cache(_OmniGeneratorSingleton._pipe.transformer)
        return ([img_tensor],)
