from base64 import b64decode, b64encode
from io import BytesIO
from typing import List

from PIL import Image

from .exception import CustomException


def is_valid_image(file_type: List[str]) -> bool:
    valid_type = ['image/png', 'image/jpeg', 'image/jpg']
    #
    return all([ftype in valid_type for ftype in file_type])


def request_to_pil(data) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert('RGB')
    except:
        raise CustomException('Cannot read image', 422)


def pil_to_base64(pil_img: Image.Image) -> str:
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_pil(base64_str: str) -> Image.Image:
    return Image.open(BytesIO(b64decode(base64_str))).convert('RGB')
