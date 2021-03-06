from typing import List, Tuple

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, cfg_path, weight_path, confidence_thresh=0.5):
        self.detector = cv2.dnn.readNetFromCaffe(cfg_path, weight_path)
        self.confidence_thresh = confidence_thresh

    @staticmethod
    def _batch_blob(images: List[np.ndarray],
                    size: Tuple[int, int],
                    mean=(104.0, 177.0, 123.0)) -> np.ndarray:
        blobs = cv2.dnn.blobFromImages(images,
                                       size=size,
                                       scalefactor=1.0,
                                       mean=mean,
                                       swapRB=False,
                                       crop=False)

        return blobs

    def batch_detect(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Detect faces in all input images.

        :param images: List of cv2-image
        :return: (original image with rectangle drawn on it, cropped face from original image)
        """

        #
        target_size = (300, 300)

        # convert to blob image
        blob_images = self._batch_blob(images, size=target_size, mean=(104.0, 177.0, 123.0))

        # feedforward
        self.detector.setInput(blob_images)
        detect_rs = self.detector.forward()

        cropped_face = []
        # take result for each images
        for img_idx, current_img in enumerate(images):

            # grab detect result for this image
            rs_idx = detect_rs[:, :, :, 0] == img_idx
            img_detect_rs = detect_rs[rs_idx, :]

            # select the most confidence box
            max_cf_idx = np.argmax(img_detect_rs[:, 2])
            (h, w) = current_img.shape[:2]
            box = img_detect_rs[max_cf_idx, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # take the score for visualize
            cf_score = img_detect_rs[max_cf_idx, 2]

            if cf_score < self.confidence_thresh:
                continue

            # grab face region
            face = current_img[start_y:end_y, start_x:end_x]
            cropped_face.append(np.copy(face))

            # draw result (scale border before draw)
            start_x, start_y = map(lambda x: max(0, x - 10), (start_x, start_y))
            end_x = min(w, end_x + 10)
            end_y = min(h, end_y + 10)
            cv2.rectangle(images[img_idx], (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)

        return images, cropped_face
