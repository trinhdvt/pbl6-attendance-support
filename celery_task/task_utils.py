import os
from base64 import b64decode, b64encode
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image
from loguru import logger


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_base64(pil_img) -> str:
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_pil(base64_str) -> Image.Image:
    return Image.open(BytesIO(b64decode(base64_str)))


async def submit_results(extracted_rs):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    default_url = "http://illusion.codes:9999/api/records"
    submit_url = os.getenv("BE_SERVER", default=default_url)

    data = {
        'examCode': extracted_rs['examCode'],
        "status": extracted_rs['match'],
        'studentId': extracted_rs['id'],
        'studentName': extracted_rs['name'],
        'className': extracted_rs['class'],
        "facultyName": extracted_rs['faculty'],
        'year': extracted_rs['year'],
        'birthday': extracted_rs['birth_day'],
        'ipAddress': extracted_rs['ipAddress'],
        'userAgent': extracted_rs['userAgent'],
        'faceImg': extracted_rs['faceImg'],
        'cardImg': extracted_rs['cardImg'],
        'cropCard': extracted_rs['cropCard']
    }
    try:
        r = requests.post(submit_url, headers=headers, data=data, timeout=5)
        if r.status_code != 200:
            logger.error(f"Submit result failed: {r.status_code}")
    except requests.exceptions.RequestException as e:
        logger.debug(f"Submit failed: {str(e)}")


async def save_log_img(img_list, filename, save_dir):
    assert len(img_list) == len(filename)

    for img, path in zip(img_list, filename):
        await save_img(img, save_dir + path)


async def save_img(img, out_path):
    cv2_to_pil(img).save(out_path, quality=85)
