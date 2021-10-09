from src.face_comparator import FaceComparator
import os

ROOT_PATH = os.path.dirname(__file__)

Cfg = {
    'face': {
        'detector': {
            'model-cfg': f"{ROOT_PATH}/resources/res10_300x300.txt",
            'model-weight': f"{ROOT_PATH}/resources/res10_300x300.caffemodel"
        },
        'alignment': {
            'model-cfg': f"{ROOT_PATH}/resources/haarcascade_eye.xml"
        },
        'embedding': {
            'model-weight': f"{ROOT_PATH}/resources/nn4.small2.v1.t7"
        }
    }
}


def load_model():
    face_comparator = FaceComparator(detector_cfg=Cfg['face']['detector']['model-cfg'],
                                     detector_weight=Cfg['face']['detector']['model-weight'],
                                     alignment_cfg=Cfg['face']['alignment']['model-cfg'],
                                     embedding_weight=Cfg['face']['embedding']['model-weight'])

    return face_comparator
