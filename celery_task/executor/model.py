import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import uvloop
from loguru import logger

from celery_task.task_utils import cv2_to_pil, pil_to_cv2, base64_to_pil, save_log_img, submit_results
from loader import load_model


class Executor:
    def __init__(self):
        self.face_compare, self.card_detector, self.card_reader, self.card_cropper = load_model()

    def process(self, task_kwargs):
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
        with ThreadPoolExecutor(max_workers=2) as executor:
            thread_1 = executor.submit(self.compare_faces, face_img, cropped_card, card_img)
            thread_2 = executor.submit(self.extract_info, *list(map(cv2_to_pil, [cropped_card, card_img])))
            #
            compare_rs = thread_1.result()
            extracted_rs = thread_2.result()

        # combine results and return
        extracted_rs.update(compare_rs)
        face_fn = task_kwargs['face-fn']
        card_fn = task_kwargs['card-fn']

        #
        end = time.time()
        extracted_rs.update({
            "request-time": end - start
        })

        # prepare data to save to disk
        file_name_to_save = [face_fn, card_fn, "crop_" + card_fn]
        file_name_to_save = list(map(lambda f: f"{int(start)}_{f}", file_name_to_save))
        img_to_save = [face_img, card_img, cropped_card]

        # prepare data to send to backend server
        file_prefix = "https://admin.illusion.codes/images"

        extracted_rs.update({
            "examCode": task_kwargs['exam_code'],
            "ipAddress": task_kwargs['remote_addr'],
            "userAgent": task_kwargs['user_agent'],
            "faceImg": file_prefix + "/" + file_name_to_save[0],
            "cardImg": file_prefix + "/" + file_name_to_save[1],
            "cropCard": file_prefix + "/" + file_name_to_save[2],
        })

        # save log and submit results
        async def post_process():
            await asyncio.gather(save_log_img(img_to_save, file_name_to_save, task_kwargs['log_dir']),
                                 submit_results(extracted_rs))

        uvloop.install()
        asyncio.run(post_process())

        #
        return extracted_rs

    def extract_info(self, card_img, raw_card_img):
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

        # ocr step - image extracted to text
        reader_rs = self.card_reader.batch_predict(extracted_rs.values(), show_time=True)

        # update extracted text for each label
        for idx, label in enumerate(extracted_rs.keys()):
            extracted_rs[label] = reader_rs[idx]

        # update results with missing info before return
        extracted_rs.update(missing_info)
        extracted_rs.update({"extract-time": time.time() - start})

        #
        return extracted_rs

    def detect_card_info(self, card_img):
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

    def crop_card(self, img, to_cv2=True):
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
        cropped_img = self.card_cropper.transform(img)

        return cropped_img

    def compare_faces(self, face_img, cropped_card_img, raw_card_img):
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
            is_match, distance = self.face_compare.predict([face_img, raw_card_img])

        #

        return {
            "match": is_match,
            "distance": distance,
            "compare-time": time.time() - start
        }
