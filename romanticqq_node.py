from .seededit_api import SEEDEDIT
from .upload_minio import MINIO_UPLOAD

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "seededit_api": SEEDEDIT,
    "upload_minio": MINIO_UPLOAD
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "seededit_api": "seededit_api",
    "upload_minio": "upload_minio"
}
