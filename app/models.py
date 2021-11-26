from pydantic import BaseModel, validator

from .app_utils import base64_to_pil
from .exception import CustomException


class Base64Input(BaseModel):
    examCode: str
    face_img: str
    card_img: str

    @validator('face_img', 'card_img')
    def must_be_base64_img(cls, v):
        try:
            base64_to_pil(v)
        except:
            raise CustomException(message="Parameter must be base64 image", status_code=400)
        return v
