import os
from typing import List, Tuple

import face_recognition as fr
import numpy as np

from .face_alignment import FaceAlignment
from .face_detector import FaceDetector
from .face_embedding import FaceEmbedding


class FaceComparator:
    def __init__(self, detector_cfg, detector_weight,
                 alignment_cfg,
                 embedding_weight):
        self.face_detector = FaceDetector(cfg_path=detector_cfg,
                                          weight_path=detector_weight)

        self.face_alignment = FaceAlignment(cfg_path=alignment_cfg)

        self.face_embedding = FaceEmbedding(weight_path=embedding_weight)

        self.threshold = float(os.getenv("FACE_THRESH", default=0.6))

    def predict(self, images: List[np.ndarray]) -> Tuple[bool, float]:
        """
        Compare face function

        :param images: List of cv2-image
        :return: A tuple of True/False which indicates that two face is the same and distance between them
        """

        # detect face
        _, cropped_face = self.face_detector.batch_detect(images)

        # align face
        for i, face in enumerate(cropped_face):
            cropped_face[i] = self.face_alignment.align(face)

        #
        return self._compare(cropped_face)

    def _compare(self, face_images: List[np.ndarray]) -> Tuple[bool, float]:
        # encoding face
        embedding = []
        for face in face_images:
            emb_vt = fr.face_encodings(face, model='large')
            if len(emb_vt) != 0:
                embedding.append(emb_vt[0])
            else:
                emb_vt = self.face_embedding.embedding([face])
                embedding.append(emb_vt[0].reshape(-1, 1).squeeze())

        # calculate euclidean distance
        distance = fr.face_distance([embedding[0]], embedding[1])

        # if distance < 0.6 then it's match otherwise no
        is_match = (distance <= self.threshold).tolist()[0]

        return is_match, distance.tolist()[0]
