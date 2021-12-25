import os
from pathlib import Path

from loguru import logger

from src.card_recognition import Detector, Reader, Cropper
from src.face_comparator import FaceComparator
from src.utils import download_model

ROOT_PATH = Path().resolve().parent.absolute()

Cfg = {
    'face': {
        'detector': {
            'cfg': f"{ROOT_PATH}/resources/res10_300x300.txt",
            'weight': f"{ROOT_PATH}/resources/res10_300x300.caffemodel",
            'drive-id': '1GuuaMWwqtzbGQBEYSr4xkJadQd93HAXh'
        },
        'alignment': {
            'cfg': f"{ROOT_PATH}/resources/haarcascade_eye.xml"
        },
        'embedding': {
            'weight': f"{ROOT_PATH}/resources/nn4.small2.v1.t7",
            'drive-id': "1JQuYjM-sep1ptf_xa9VMoyAyrQHbCEl0"
        }
    },
    'card': {
        'detector': {
            'weight': f"{ROOT_PATH}/resources/yolov5_300.pt",
            'drive-id': "1S2-Dq-6OTikXznkMkTE6M4aa-LYoTz2A"
        },
        "reader": {
            'cfg': f"{ROOT_PATH}/resources/reader_cfg.yml",
            "weight": f"{ROOT_PATH}/resources/reader_transformer.pth",
            "drive-id": "1vVpXlkkpwl0Ac-Ht_C464ENd8SHP1WIs"
        }
    }
}


def load_model():
    #
    model_check()

    #
    face_comparator = FaceComparator(detector_cfg=Cfg['face']['detector']['cfg'],
                                     detector_weight=Cfg['face']['detector']['weight'],
                                     alignment_cfg=Cfg['face']['alignment']['cfg'],
                                     embedding_weight=Cfg['face']['embedding']['weight'])

    card_detector = Detector(weight_path=Cfg['card']['detector']['weight'])

    card_reader = Reader(cfg_path=Cfg['card']['reader']['cfg'],
                         weight_path=Cfg['card']['reader']['weight'])

    card_cropper = Cropper()

    return face_comparator, card_detector, card_reader, card_cropper


def model_check():
    logger.info("Begin model check")

    for keys in Cfg.values():
        for model in keys.values():
            #
            if 'weight' in model.keys():
                weight_path = model['weight']
                #
                if not os.path.isfile(weight_path):
                    try:
                        drive_id = model['drive-id']
                        download_model(drive_id, weight_path)
                    except Exception as e:
                        logger.error(str(e))
                        raise e

    logger.info("Model check done!")
