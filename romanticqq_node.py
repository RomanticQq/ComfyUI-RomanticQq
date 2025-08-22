from .seededit_api import SEEDEDIT
from .upload_minio import MINIO_UPLOAD
from .qwenvl_omni_api import QWENVL_Omni
from .qwenvl_api import QWENVL
from .jimeng_depth_2_pro_api import Jimeng_Depth_2_Pro
from .jimeng_depth_xl_api import Jimeng_Depth_Xl
from .jimeng_role_api import Jimeng_Role
from .omniconsistency_nodes import Comfyui_OmniConsistency
from .fluxgym_caption import FluxGymCaption
from .add_two_image import AddTwoImage
from .add_text import AddText
from .word_title import WordTitle
from .split_filter_concat import SplitFilterConcat
from .str_lower import StrLower
from .lora_path import LoraPath
from .add_vertical_text import ADD_VERTICAL_TEXT
from .get_width_height_region import GetWidthHeightRegion
from .color_to_color import ColorToColor
from .color_to_image import ColorToImage
from .color_list_to_image import ColorListToImage
from .padding_image import PaddingImage

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "seededit_api": SEEDEDIT,
    "upload_minio": MINIO_UPLOAD,
    "qwenvl_omni_api": QWENVL_Omni,
    "qwenvl_api": QWENVL,
    "jimeng_role_api": Jimeng_Role,
    "jimeng_depth_xl_api": Jimeng_Depth_Xl,
    "jimeng_depth_2_pro_api": Jimeng_Depth_2_Pro,
    "Comfyui_OmniConsistency_fq": Comfyui_OmniConsistency,
    "fluxgym_caption": FluxGymCaption,
    "add_two_image": AddTwoImage,
    "add_text": AddText,
    "word_title": WordTitle,
    "split_filter_concat": SplitFilterConcat,
    "str_lower": StrLower,
    "lora_path": LoraPath,
    "add_vertical_text": ADD_VERTICAL_TEXT,
    "get_width_height_region": GetWidthHeightRegion,
    "color_to_color": ColorToColor,
    "color_to_image": ColorToImage,
    "color_list_to_image": ColorListToImage,
    "padding_image": PaddingImage,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "seededit_api": "seededit_api",
    "upload_minio": "upload_minio",
    "qwenvl_omni_api": "qwenvl_omni_api",
    "qwenvl_api": "qwenvl_api",
    "jimeng_role_api": "jimeng_role_api",
    "jimeng_depth_xl_api": "jimeng_depth_xl_api",
    "jimeng_depth_2_pro_api": "jimeng_depth_2_pro_api",
    "Comfyui_OmniConsistency_fq": "OmniConsistency-Generator-fq",
    "fluxgym_caption": "fluxgym_caption",
    "add_two_image": "add_two_image",
    "add_text": "add_text",
    "word_title": "word_title",
    "split_filter_concat": "split_filter_concat",
    "str_lower": "str_lower",
    "lora_path": "lora_path",
    "add_vertical_text": "add_vertical_text",
    "get_width_height_region": "get_width_height_region",
    "color_to_color": "color_to_color",
    "color_to_image": "color_to_image",
    "color_list_to_image": "color_list_to_image",
    "padding_image": "padding_image",
}