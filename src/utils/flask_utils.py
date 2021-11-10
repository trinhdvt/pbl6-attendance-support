import os

import requests
import time

from src.utils.image_utils import cv2_to_pil

headers = {'Content-Type': 'application/x-www-form-urlencoded'}
default_url = "http://illusion.codes:9999/api/records"
submit_url = os.getenv("BE_SERVER", default=default_url)


def submit_results(extracted_rs):
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
    r = requests.post(submit_url, headers=headers, data=data)
    assert r.status_code == 200


def save_log_img(img_list, filename, save_dir):
    assert len(img_list) == len(filename)

    for img, path in zip(img_list, filename):
        cv2_to_pil(img).save(save_dir + path, quality=85)
        time.sleep(0.001)
