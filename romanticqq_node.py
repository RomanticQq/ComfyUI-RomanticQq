from .seededit_api import SEEDEDIT
from .upload_minio import MINIO_UPLOAD
from .qwenvl_omni_api import QWENVL_Omni
from .qwenvl_api import QWENVL
from .jimeng_depth_2_pro_api import Jimeng_Depth_2_Pro
from .jimeng_depth_xl_api import Jimeng_Depth_Xl
from .jimeng_role_api import Jimeng_Role
from .groundingdino_cut import GroundingDino
from .grounded_sam2_cut_gaussian import GroundedSam2CutGaussian
from .omniconsistency_nodes import Comfyui_OmniConsistency

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
    "groundingdino": GroundingDino,
    "grounded_sam2_cut_gaussian": GroundedSam2CutGaussian,
    "Comfyui_OmniConsistency_fq": Comfyui_OmniConsistency,
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
    "groundingdino": "GroundingDino",
    "grounded_sam2_cut_gaussian": "GroundedSam2CutGaussian",
    "Comfyui_OmniConsistency_fq": "OmniConsistency-Generator-fq",
}