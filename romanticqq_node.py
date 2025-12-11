from .api.jimeng.seededit_api import SEEDEDIT
from .api.upload.upload_minio import MINIO_UPLOAD
from .api.jimeng.jimeng_depth_2_pro_api import Jimeng_Depth_2_Pro
from .api.jimeng.jimeng_depth_xl_api import Jimeng_Depth_Xl
from .api.jimeng.jimeng_role_api import Jimeng_Role
from .prompt.fluxgym_caption import FluxGymCaption
from .image.add_two_image import AddTwoImage
from .text.add_text import AddText
from .text.word_title import WordTitle
from .text.split_filter_concat import SplitFilterConcat
from .text.str_lower import StrLower
from .lora_path import LoraPath
from .text.add_vertical_text import ADD_VERTICAL_TEXT
from .image.get_width_height_region import GetWidthHeightRegion
from .color.color_to_color import ColorToColor
from .color.color_to_image import ColorToImage
from .color.color_list_to_image import ColorListToImage
from .image.padding_image import PaddingImage
from .text.random_text import RANDOM_TEXT
from .text.text_translate import Text_Translation
from .text.random_text_v2 import RANDOM_TEXT_V2
from .api.custom_api import CustomAPI
from .api.aigc.wordcard_api import WORDCARD
from .api.jimeng.jimeng_t2i_3 import JIMENG_T2I_3
from .text.text_json import TEXT_JSON
from .text.text_kv_get_value import TEXT_KV_GET_VALUE
from .api.tuzi.gemini_image_api import GeminiImageAPI
from .api.tuzi.gemini_image_official_api import GeminiImageOfficialAPI
from .api.jieyue.jieyue_detect_word import JIeyueDetectWordAPI
from .api.jieyue.jieyue_detect_draw_bbox import JIeyueDetectDrawBboxAPI
from .text.florence2_get_word import Florence2GetWord      
from .text.text_json_catroon import TEXT_JSON_CATROON
from .text.text_kv_to_json import TEXT_KV_TO_JSON
from .image.image_ratio import ImageRatio
from .number.padding_size import PaddingSize
from .tarot.tarot_slect import tarot_slect
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "seededit_api": SEEDEDIT,
    "upload_minio": MINIO_UPLOAD,
    "jimeng_role_api": Jimeng_Role,
    "jimeng_depth_xl_api": Jimeng_Depth_Xl,
    "jimeng_depth_2_pro_api": Jimeng_Depth_2_Pro,
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
    "random_text": RANDOM_TEXT,
    "text_translation": Text_Translation,
    "random_text_v2": RANDOM_TEXT_V2,
    "custom_api": CustomAPI,
    "wordcard_api": WORDCARD,
    "jimeng_t2i_3": JIMENG_T2I_3,
    "text_json": TEXT_JSON,
    "text_kv_get_value": TEXT_KV_GET_VALUE,
    "gemini_image_api": GeminiImageAPI,
    "gemini_image_official_api": GeminiImageOfficialAPI,
    "jieyue_detect_word_api": JIeyueDetectWordAPI,
    "jieyue_detect_draw_bbox_api": JIeyueDetectDrawBboxAPI,
    "florence2_get_word": Florence2GetWord,
    "text_json_catroon": TEXT_JSON_CATROON,
    "text_kv_to_json": TEXT_KV_TO_JSON,
    "image_ratio": ImageRatio,
    "padding_size": PaddingSize,
    "tarot_slect": tarot_slect,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "seededit_api": "seededit_api",
    "upload_minio": "upload_minio",
    "jimeng_role_api": "jimeng_role_api",
    "jimeng_depth_xl_api": "jimeng_depth_xl_api",
    "jimeng_depth_2_pro_api": "jimeng_depth_2_pro_api",
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
    "random_text": "random_text",
    "text_translation": "text_translation",
    "random_text_v2": "random_text_v2",
    "custom_api": "custom_api",
    "wordcard_api": "wordcard_api",
    "jimeng_t2i_3": "jimeng_t2i_3",
    "text_json": "text_json",
    "text_kv_get_value": "text_kv_get_value",
    "gemini_image_api": "gemini_image_api",
    "gemini_image_official_api": "gemini_image_official_api",
    "jieyue_detect_word_api": "jieyue_detect_word_api",
    "jieyue_detect_draw_bbox_api": "jieyue_detect_draw_bbox_api",
    "florence2_get_word": "florence2_get_word",
    "text_json_catroon": "text_json_catroon",
    "text_kv_to_json": "text_kv_to_json",
    "image_ratio": "image_ratio",
    "padding_size": "padding_size",
    "tarot_slect": "tarot_slect",
}