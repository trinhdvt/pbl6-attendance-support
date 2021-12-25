from base64 import b64decode, b64encode
from io import BytesIO
from typing import Union, Any

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(img: Union[Image.Image, Any]) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')


def pil_to_base64(pil_img: Image.Image) -> str:
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_pil(base64_str: str) -> Image.Image:
    return Image.open(BytesIO(b64decode(base64_str))).convert('RGB')
