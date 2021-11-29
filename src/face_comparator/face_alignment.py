from typing import Union, Iterable

import cv2
import numpy as np
from PIL import Image
from loguru import logger


class FaceAlignment:
    def __init__(self, cfg_path):
        """
        Face Alignment class implemented by OpenCV CascadeClassifier
        """

        self.eye_cascade = cv2.CascadeClassifier(cfg_path)

    def align(self, img: np.ndarray) -> np.ndarray:
        """
        Align a single face

        :param img: an opencv-image containing a single face need to be aligned
        :return: aligned face or original face if it cannot be aligned
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)

        if len(eyes) == 2:
            left_eye, right_eye = sorted(eyes, key=lambda x: x[0])

            # center of eyes
            # index base: 0-x, 1-y, 2-w, 3-h
            left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
            left_eye_x = left_eye_center[0]
            left_eye_y = left_eye_center[1]

            right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
            right_eye_x = right_eye_center[0]
            right_eye_y = right_eye_center[1]

            # find rotation direction
            if left_eye_y > right_eye_y:
                point_3rd = (right_eye_x, left_eye_y)
                direction = -1  # rotate same direction to clock
            else:
                point_3rd = (left_eye_x, right_eye_y)
                direction = 1  # rotate inverse direction of clock

            # calc rotate angle
            a = self._euclidean_distance(left_eye_center, point_3rd)
            b = self._euclidean_distance(right_eye_center, point_3rd)
            c = self._euclidean_distance(right_eye_center, left_eye_center)

            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)
            angle = (angle * 180) / np.pi
            if direction == -1:
                angle = 90 - angle

            # rotate image
            new_img = Image.fromarray(img)
            new_img = np.array(new_img.rotate(direction * angle))

            return new_img

        logger.debug("Not thing to align! Return the origin")
        return img

    @staticmethod
    def _euclidean_distance(a: Iterable, b: Iterable) -> Union[float, np.ndarray]:
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)
