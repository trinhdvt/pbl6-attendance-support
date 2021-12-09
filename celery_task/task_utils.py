import asyncio
import os
from base64 import b64decode, b64encode
from io import BytesIO
from typing import Dict, Optional, List, Union, Any

import cv2
import numpy as np
import requests
from PIL import Image
from loguru import logger


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


async def submit_results(extracted_rs: Dict[str, Optional[str]]):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    default_url = "http://illusion.codes:9999"
    submit_url = os.getenv("BE_SERVER", default=default_url)

    data = {
        'examCode': extracted_rs['examCode'],
        "status": extracted_rs.get('match', 0),
        'studentId': extracted_rs.get('id', None),
        'studentName': extracted_rs.get('name', None),
        'className': extracted_rs.get('class', None),
        "facultyName": extracted_rs.get('faculty', None),
        'year': extracted_rs.get('year', None),
        'birthday': extracted_rs.get('birth_day', None),
        'ipAddress': extracted_rs['ipAddress'],
        'userAgent': extracted_rs['userAgent'],
        'faceImg': extracted_rs['faceImg'],
        'cardImg': extracted_rs['cardImg'],
        'cropCard': extracted_rs['cropCard']
    }
    try:
        r = requests.post(f"{submit_url}/api/records", headers=headers, data=data, timeout=5)
        if r.status_code != 200:
            logger.error(f"Submit result failed: {r.content}")
    except requests.exceptions.RequestException as e:
        logger.debug(f"Submit failed: {str(e)}")


async def save_log_img(img_list: List[np.ndarray],
                       filename: List[str],
                       save_dir: str):
    if len(img_list) != len(filename):
        logger.error(f"img_list and filename length not equal: {len(img_list)} vs {len(filename)}")
        return

    async def save_img(img_: np.ndarray,
                       out_path: str):
        pil_image = cv2_to_pil(img_)
        pil_image.thumbnail((600, 600), Image.ANTIALIAS)
        pil_image.save(out_path, quality=85)

    task_list = [save_img(img, save_dir + path) for img, path in zip(img_list, filename)]
    await asyncio.gather(*task_list)


async def get_backup_image(student_id: str) -> Optional[Image.Image]:
    base_url = os.getenv("BACKUP_IMG_URL", default="https://pbl6-sv-img.s3.ap-southeast-1.amazonaws.com")
    img_url = f"{base_url}/{student_id}.jpg"
    #
    response = requests.get(img_url, stream=True)
    if response.status_code == 200:
        return Image.open(response.raw).convert('RGB')
    else:
        return None
