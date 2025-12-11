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
import sys
import json

# Ensure the parent directory (which contains the `tarot` package) is on sys.path.
# The previous code used split('tarot_get.py') which doesn't match this file and
# could end up appending a file path instead of the correct directory — that
# prevents Python from finding the `tarot` package. Add the parent directory
# to sys.path so `from tarot.tarot_data import ...` works reliably.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from tarot.tarot_data import card_array_num_kv, card_direction, tarot_cards, card_array_position_kv, card_array_postion_meaning_kv


class tarot_slect:
    def __init__(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
    @classmethod
    def INPUT_TYPES(s):
        card_array = ['single_tarot', 'metaphors_of_time', 'tarot_cross_spread']
        return {
            "required": {
                "card_array": (card_array,),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "step": 1, "control_after_generate": True}),

            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq/tarot"
    def test(self, card_array, random_seed):
        np.random.seed(random_seed)
        select_num = card_array_num_kv[card_array]
        selected_cards = np.random.choice(tarot_cards, size=select_num, replace=False)
        select_positions = np.random.choice(card_direction, size=select_num)  # 随机选择正位或逆位
        tarot_symbolism = json.load(open(os.path.join(os.path.dirname(__file__).split('/tarot/')[0], "tarot_symbolism.json"), "r", encoding='utf-8'))

        res_list = [{"name":select_card, "postion":select_position, "symbolism": tarot_symbolism[select_card][select_position],"class": tarot_symbolism[select_card]["class"], "card_array_position": card_array_position, "card_array_postion_meaning": card_array_postion_meaning} for select_card, select_position, card_array_position, card_array_postion_meaning in zip(selected_cards, select_positions, card_array_position_kv[card_array], card_array_postion_meaning_kv[card_array])]
        res_kv = {}
        res_kv["card_array"] = card_array
        res_kv["select_num"] = select_num
        res_kv["cards"] = res_list
        res = json.dumps(res_kv, ensure_ascii=False, indent=4)
        return (res,)