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

class SplitFilterConcat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING",),
                "split": ("STRING",{"default": None}),
                "filter": ("STRING",{"default": None}),
                "concat": ("STRING",{"default": None}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    def test(self, text, split=None, filter=None, concat=None):
        print("text: ", text)
        text = text.strip()
        print("strip text: ", text)
        res_text = ""
        try:
            if split is None or filter is None or concat is None:
                res_text = text
            else:
                filter = filter.split(",") if filter else None
                if isinstance(filter, str):
                    filter = [filter]
                split_texts = text.split(split)
                print("split_texts: ", split_texts)
                new_texts = []
                for t in split_texts:
                    t = t.strip()
                    flag = True
                    for f in filter:
                        if f in t or t.strip() == "":
                            flag = False
                            break
                    if flag:
                        new_texts.append(t)
                print("new_texts: ", new_texts)
                res_text = concat.join(new_texts)+concat
            res_text = res_text.strip()
            print("res_text: ", res_text)
        except Exception as e:
            print(e)

        return (res_text,)