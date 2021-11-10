import cv2
import numpy as np


class FaceDetector:
    def __init__(self, cfg_path, weight_path, confidence_thresh=0.5):

        self.detector = cv2.dnn.readNetFromCaffe(cfg_path, weight_path)
        self.confidence_thresh = confidence_thresh

    @staticmethod
    def _batch_blob(images, size, scale_factor=1.0, mean=(0, 0, 0)):
        blobs = cv2.dnn.blobFromImages(images,
                                       size=size,
                                       scalefactor=scale_factor,
                                       mean=mean,
                                       swapRB=False,
                                       crop=False)

        return blobs

    def batch_detect(self, images):
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
            label = f"{cf_score * 100:.2f}"
            y = start_y - 10 if start_y > 20 else start_y + 10

            if cf_score < self.confidence_thresh:
                continue

            # grab face region
            face = current_img[start_y:end_y, start_x:end_x]
            cropped_face.append(np.copy(face))

            # draw result
            # cv2.rectangle(images[img_idx], (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            # cv2.putText(images[img_idx], label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return images, cropped_face
