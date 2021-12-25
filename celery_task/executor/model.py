import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

import uvloop

from celery_task.task_exception import TaskException
from celery_task.task_utils import *
from loader import load_model


class Executor:
    def __init__(self):
        self.face_compare, self.card_detector, self.card_reader, self.card_cropper = load_model()
        uvloop.install()

    def process(self, task_kwargs: Dict[str, str]) -> Dict[str, Any]:
        """
        Entry point for processing task

        :param task_kwargs: Task parameters
        :return: Result as a dictionary
        """

        start = time.time()
        #
        face_img = base64_to_pil(task_kwargs['face-img-b64'])
        card_img = base64_to_pil(task_kwargs['card-img-b64'])
        #
        face_img, card_img = list(map(pil_to_cv2, [face_img, card_img]))
        # crop card
        cropped_card = self.crop_card(card_img)

        # execute in 2-thread
        exception = None
        compare_rs, extracted_rs = {}, {}
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                thread_1 = executor.submit(self.compare_faces, face_img, cropped_card, card_img)
                thread_2 = executor.submit(self.extract_info, *list(map(cv2_to_pil, [cropped_card, card_img])))
                #
                compare_rs = thread_1.result()
                extracted_rs = thread_2.result()
        #
        except Exception as e:
            # catch exception then raise it later
            exception = e
        finally:
            # combine results
            extracted_rs.update(compare_rs)

            # save additional info
            extracted_rs.update({
                "request-time": time.time() - start,
                "cropCard": pil_to_base64(cv2_to_pil(cropped_card))
            })

        # if error occurred, raise it
        if exception:
            raise exception
        #
        return extracted_rs

    def extract_info(self,
                     card_img: Image,
                     raw_card_img: Image) -> Dict[str, Any]:
        """
        Recognize info in student's card

        :param card_img: Cropped card image (PIL image)
        :param raw_card_img: Raw card image (PIL image)
        :return: A dictionary representing extracted information
        """

        start = time.time()

        # detect information bounding box
        missing_info, extracted_rs = self.detect_card_info(card_img)

        # retry to detect on raw image
        if len(missing_info.keys()) >= 2 or 'id' in missing_info.keys():
            missing_info, extracted_rs = self.detect_card_info(raw_card_img)

            # raise exception if still missing
            if len(missing_info.keys()) >= 2 or 'id' in missing_info.keys():
                raise TaskException("Failed to detect info in card!")

        # ocr step - image extracted to text
        reader_rs = self.card_reader.batch_predict(extracted_rs.values())

        # update extracted text for each label
        extracted_rs = {label: reader_rs[idx] for idx, label in enumerate(extracted_rs.keys())}

        # update results with missing info before return
        extracted_rs.update(missing_info)
        extracted_rs.update({"extract-time": time.time() - start})

        #
        return extracted_rs

    def detect_card_info(self, card_img: Image) -> Tuple[Dict[str, None], Dict[str, Image]]:
        """
        Detect info bounding_box in student's card and crop it into PIL image

        :param card_img: Cropped card image (PIL image)
        :return: Missing class and cropped class
        """

        # detect information bounding box
        detected_rs = self.card_detector.detect_crop(card_img, save_file=False, verbose=True)

        # check for missing information before goto next step
        missing_info = {}
        extracted_rs = {}
        for label, cropped_img in detected_rs.items():
            if cropped_img is None:
                missing_info[label] = None
            else:
                extracted_rs[label] = cropped_img

        return missing_info, extracted_rs

    def crop_card(self, img: np.ndarray) -> np.ndarray:
        """
        Crop student's card from input image

        :param img: Cv2-image
        :return: Cropped image (cv2-image)
        """

        # crop
        return self.card_cropper.transform(img)

    def compare_faces(self,
                      face_img: np.ndarray,
                      cropped_card_img: np.ndarray,
                      raw_card_img: np.ndarray) -> Dict[str, Union[bool, float]]:
        """
        Compare two faces in two images

        :param face_img: Raw face image (cv2-image)
        :param cropped_card_img: Cropped card image (cv2-image)
        :param raw_card_img: Raw card image (cv2-image)
        :return: is_match and distance
        """

        start = time.time()
        try:
            is_match, distance = self.face_compare.predict([face_img, cropped_card_img])
        except Exception:
            # retry if cannot detect faces from card_img
            try:
                is_match, distance = self.face_compare.predict([face_img, raw_card_img])
            except Exception:
                raise TaskException("Cannot detect faces!")

        #
        return {
            "match": is_match,
            "distance": distance,
            "compare-time": time.time() - start
        }
