import cv2
import numpy as np


class FaceEmbedding:
    def __init__(self, weight_path):
        self.encoder = cv2.dnn.readNetFromTorch(weight_path)

    @staticmethod
    def _pre_process(images):
        blob = cv2.dnn.blobFromImages(images,
                                      size=(96, 96),
                                      scalefactor=1.0 / 255,
                                      mean=(0, 0, 0),
                                      swapRB=False,
                                      crop=False)
        return blob

    def embedding(self, images):
        """
        Calculate embedding for list of images.

        :param images: List of OpenCV images
        :return: list of embedding vectors for each image
        """
        # preprocess
        blob_img = self._pre_process(images)  # shape Nx3x96x96

        # feedforward
        self.encoder.setInput(blob_img)
        emb_vectors = self.encoder.forward()  # shape Nx128

        #
        emb_vectors = np.expand_dims(emb_vectors, axis=1)  # shape Nx1x128
        return emb_vectors
