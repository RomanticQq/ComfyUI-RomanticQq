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
                "filter": ("LIST",{"default": None}),
                "concat": ("STRING",{"default": None}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "RomanticQq"
    def test(self, text, split=None, flter=None, concat=None):
        print("text: ", text)
        try:
            if split is None or filter is None or concat is None:
                res_text = text
            else:
                split_texts = text.split(split)
                new_texts = []
                for t in split_texts:
                    if flter is not None and t in flter:
                        continue
                    new_texts.append(t)
                res_text = concat.join(new_texts)+concat
            print("res_text: ", res_text)
        except Exception as e:
            print(e)

        return (res_text,)