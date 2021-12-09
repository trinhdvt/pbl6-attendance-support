import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Union, Tuple, Any

import numpy as np
import uvloop
from PIL.Image import Image
from loguru import logger

from celery_task.task_exception import TaskException
from celery_task.task_utils import cv2_to_pil, pil_to_cv2, base64_to_pil, save_log_img, submit_results, get_backup_image
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
        cropped_card = self.crop_card(card_img, to_cv2=False)
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

            # retry to detect on backup image
            if compare_rs['match'] is False:
                student_id = extracted_rs['id']
                logger.debug(f"Retrying to compare faces with student id {student_id}")
                #
                backup_img = asyncio.run(get_backup_image(student_id))
                if backup_img is None:
                    logger.debug(f"No backup image for {student_id} found!")
                else:
                    backup_img = pil_to_cv2(backup_img)
                    is_match, distance = self.face_compare.predict([face_img, backup_img])
                    compare_rs['match'] = is_match
                    compare_rs['distance'] = distance
        except Exception as e:
            # catch exception then raise it later
            exception = e
        finally:
            # combine results
            extracted_rs.update(compare_rs)
            #
            extracted_rs.update({"request-time": time.time() - start})
            # prepare data to save
            face_fn = task_kwargs['face-fn']
            card_fn = task_kwargs['card-fn']
            file_name_to_save = [face_fn, card_fn, "crop_" + card_fn]
            file_name_to_save = list(map(lambda f: f"{int(start)}_{f}", file_name_to_save))
            img_to_save = [face_img, card_img, cropped_card]
            file_prefix = "https://admin.illusion.codes/images"
            #
            extracted_rs.update({
                "examCode": task_kwargs['exam_code'],
                "checkAt": task_kwargs['check_at'],
                "ipAddress": task_kwargs['remote_addr'],
                "userAgent": task_kwargs['user_agent'],
                "faceImg": file_prefix + "/" + file_name_to_save[0],
                "cardImg": file_prefix + "/" + file_name_to_save[1],
                "cropCard": file_prefix + "/" + file_name_to_save[2],
            })

            # save log and submit results
            asyncio.run(self.post_process(img_to_save, file_name_to_save, task_kwargs['log_dir'], extracted_rs))

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
            logger.debug(f"Retrying to detect missing class {missing_info.keys()}")
            missing_info, extracted_rs = self.detect_card_info(raw_card_img)

            # raise exception if still missing
            if len(missing_info.keys()) >= 2 or 'id' in missing_info.keys():
                logger.error("Failed to detect info in card!")
                raise TaskException("Không thể nhận diện thông tin trong thẻ!")

        # ocr step - image extracted to text
        reader_rs = self.card_reader.batch_predict(extracted_rs.values(), show_time=True)

        # update extracted text for each label
        extracted_rs = {label: reader_rs[idx] for idx, label in enumerate(extracted_rs.keys())}

        # update results with missing info before return
        extracted_rs.update(missing_info)
        extracted_rs.update({"extract-time": time.time() - start})

        #
        return extracted_rs

    def detect_card_info(self,
                         card_img: Image) -> Tuple[Dict[str, None], Dict[str, Image]]:
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

    def crop_card(self,
                  img: Union[np.ndarray, Image],
                  to_cv2=True) -> np.ndarray:
        """
        Crop student's card from input image

        :param to_cv2: True if input's need to be converted to cv2-image
        :param img: PIL Image
        :return: Cropped image (cv2-image)
        """

        # convert to cv2-image
        if to_cv2:
            img = pil_to_cv2(img)
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
            logger.warning("Retrying to detect faces from raw_card_img")
            try:
                is_match, distance = self.face_compare.predict([face_img, raw_card_img])
            except Exception:
                logger.error("Cannot detect faces")
                raise TaskException(message="Không thể phát hiện khuôn mặt trên ảnh!")

        #
        return {
            "match": is_match,
            "distance": distance,
            "compare-time": time.time() - start
        }

    @staticmethod
    async def post_process(img_to_save, file_name_to_save, log_dir, extracted_rs):
        await asyncio.gather(save_log_img(img_to_save, file_name_to_save, log_dir),
                             submit_results(extracted_rs))
