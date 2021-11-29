from fastapi import status
from pydantic import BaseModel, validator

from .app_utils import base64_to_pil
from .exception import CustomException


class Base64Input(BaseModel):
    examCode: str
    face_img: str
    card_img: str

    @validator('face_img', 'card_img')
    def base64_image_check(cls, v: str):
        # request's type check
        support_type = ["image/jpeg", "image/png"]
        prefix = tuple(f"data:{_};base64," for _ in support_type)
        if not v.startswith(prefix):
            raise CustomException(message='Parameter must be JPEG or PNG image',
                                  status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

        # valid base64 image check
        image_data = v.split(',')[1]
        try:
            base64_to_pil(image_data)
        except:
            raise CustomException(message="Parameter must be base64 image",
                                  status_code=status.HTTP_400_BAD_REQUEST)

        # return without base64 prefix
        return image_data
