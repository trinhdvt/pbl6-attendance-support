import os
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
from loguru import logger

from loader import load_model
from src.utils.image_utils import cv2_to_pil, pil_to_cv2
from src.utils.flask_utils import submit_results, save_log_img

face_compare, card_detector, card_reader, card_cropper = load_model()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_EXTENSIONS'] = ['jpg', 'jpeg', 'png']
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.config['LOG_DIR'] = os.getcwd() + "/upload/"


@app.route("/api/test", methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({'message': 'OK'}), 200


@app.route("/api/face-compare", methods=['POST'])
@cross_origin()
def run_for_my_life():
    """
    Compare two faces in two image API

    :return: Matching result and distance between two faces
    """

    # request parameters check
    if 'face-img' not in request.files \
            or 'card-img' not in request.files:
        resp = {"message": "Missing parameters"}
        return make_response(jsonify(resp), 400)
    #
    face_img_file = request.files['face-img']
    card_img_file = request.files['card-img']
    if not allowed_file([face_img_file.filename, card_img_file.filename]):
        resp = {"message": "Not allowed!"}
        return make_response(jsonify(resp), 400)

    # read and convert to cv2-image
    face_img, card_img = read_request_file([face_img_file, card_img_file], cvt_fn=pil_to_cv2)

    # crop card
    cropped_card = crop_card(card_img, to_cv2=False)

    # detect and compare two faces
    compare_rs = compare_faces(face_img, cropped_card, card_img)

    #
    logger.debug(f"Request ({face_img_file.filename},{card_img_file.filename}) done with {compare_rs}!")
    return jsonify(compare_rs), 200


@app.route("/api/card-recognize", methods=['POST'])
def run_for_your_life():
    """
    Card's info extract API

    :return: Extracted information as a dictionary
    """

    # parameters check
    if 'card-img' not in request.files:
        resp = {"message": "Missing parameters"}
        return make_response(jsonify(resp), 400)

    # read parameters from request
    img_file = request.files['card-img']
    [origin_img] = read_request_file([img_file], cvt_fn=pil_to_cv2)

    # crop image if need
    is_cropped = request.form.get("cropped")
    if not is_cropped:
        card_img = crop_card(origin_img, to_cv2=False)
    else:
        card_img = origin_img

    # convert to PIL Image
    # extract info
    extracted_rs = extract_info(*list(map(cv2_to_pil, [card_img, origin_img])))

    #
    return jsonify(extracted_rs), 200


@app.route("/api/check", methods=['POST'])
def run_for_our_life():
    """
    Receive face image and card image. Compare face in two image and extract card's information

    :return: Face compare result and extracted information
    """

    start = time.time()
    exam_code = request.form.get("examCode", type=str)
    # request parameters check
    if 'face-img' not in request.files \
            or 'card-img' not in request.files \
            or not exam_code:
        resp = {"message": "Missing parameters"}
        return make_response(jsonify(resp), 400)
    #
    face_img_file = request.files['face-img']
    card_img_file = request.files['card-img']
    if not allowed_file([face_img_file.filename, card_img_file.filename]):
        resp = {"message": "Not allowed!"}
        return make_response(jsonify(resp), 400)

    # read and convert to cv2-image
    face_img, card_img = read_request_file([face_img_file, card_img_file], cvt_fn=pil_to_cv2)

    # crop card
    cropped_card = crop_card(card_img, to_cv2=False)

    # execute in 2-thread
    with ThreadPoolExecutor(max_workers=2) as executor:
        thread_1 = executor.submit(compare_faces, face_img, cropped_card, card_img)
        thread_2 = executor.submit(extract_info, *list(map(cv2_to_pil, [cropped_card, card_img])))

        compare_rs = thread_1.result()
        extracted_rs = thread_2.result()

    # combine results and return
    extracted_rs.update(compare_rs)
    logger.debug(f"Request ({face_img_file.filename},{card_img_file.filename}) done with {extracted_rs}!")

    #
    end = time.time()
    extracted_rs.update({
        "request-time": end - start
    })

    # prepare data to save to disk
    file_name_to_save = [face_img_file.filename, card_img_file.filename, "crop_" + card_img_file.filename]
    file_name_to_save = list(map(lambda f: f"{int(start)}_{f}", file_name_to_save))
    img_to_save = [face_img, card_img, cropped_card]

    # prepare data to send to backend server
    file_prefix = "https://admin.illusion.codes/images"
    extracted_rs.update({
        "examCode": exam_code,
        "ipAddress": request.remote_addr,
        "userAgent": request.user_agent.browser,
        "faceImg": file_prefix + "/" + file_name_to_save[0],
        "cardImg": file_prefix + "/" + file_name_to_save[1],
        "cropCard": file_prefix + "/" + file_name_to_save[2],
    })

    # save log and submit results
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(save_log_img, img_to_save, file_name_to_save, app.config['LOG_DIR'])
        executor.submit(submit_results, extracted_rs)

    #
    return jsonify(extracted_rs), 200


def compare_faces(face_img, cropped_card_img, raw_card_img):
    """
    Compare two faces in two images

    :param face_img: Raw face image (cv2-image)
    :param cropped_card_img: Cropped card image (cv2-image)
    :param raw_card_img: Raw card image (cv2-image)
    :return: is_match and distance
    """

    start = time.time()

    try:
        is_match, distance = face_compare.predict([face_img, cropped_card_img])

    except Exception:
        # retry if cannot detect faces from card_img
        logger.warning("Retrying to detect faces from raw_card_img")
        is_match, distance = face_compare.predict([face_img, raw_card_img])

    #

    return {
        "match": is_match,
        "distance": distance,
        "compare-time": time.time() - start
    }


def crop_card(img, to_cv2=True):
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
    cropped_img = card_cropper.transform(img)

    return cropped_img


def detect_card_info(card_img):
    """
    Detect info bounding_box in student's card and crop it into PIL image

    :param card_img: Cropped card image (PIL image)
    :return: Missing class and cropped class
    """

    # detect information bounding box
    detected_rs = card_detector.detect_crop(card_img, save_file=True, verbose=True)

    # check for missing information before goto next step
    missing_info = {}
    extracted_rs = {}
    for label, cropped_img in detected_rs.items():
        if cropped_img is None:
            missing_info[label] = None
        else:
            extracted_rs[label] = cropped_img

    return missing_info, extracted_rs


def extract_info(card_img, raw_card_img):
    """
    Recognize info in student's card

    :param card_img: Cropped card image (PIL image)
    :param raw_card_img: Raw card image (PIL image)
    :return: A dictionary representing extracted information
    """

    start = time.time()

    # detect information bounding box
    missing_info, extracted_rs = detect_card_info(card_img)

    # retry to detect on raw image
    if len(missing_info.keys()) >= 2 or 'id' in missing_info.keys():
        logger.debug(f"Retrying to detect missing class {missing_info.keys()}")
        missing_info, extracted_rs = detect_card_info(raw_card_img)

    # ocr step - image extracted to text
    reader_rs = card_reader.batch_predict(extracted_rs.values(), show_time=True)

    # update extracted text for each label
    for idx, label in enumerate(extracted_rs.keys()):
        extracted_rs[label] = reader_rs[idx]

    # update results with missing info before return
    extracted_rs.update(missing_info)
    extracted_rs.update({"extract-time": time.time() - start})

    #
    return extracted_rs


def read_request_file(img_files, cvt_fn=None):
    """
    Read image from requested file

    :param img_files: List of files to read
    :param cvt_fn: Convert function (to PIL or to cv2)
    :return: List of PIL images or cv2-images
    """

    img_arr = []
    for img_file in img_files:
        img = Image.open(img_file).convert('RGB')
        if cvt_fn is not None:
            img = cvt_fn(img)
        img_arr.append(img)

    return img_arr


def allowed_file(filenames):
    return all(["." in filename and
                filename.rsplit(".", 1)[1].lower() in app.config['UPLOAD_EXTENSIONS']
                for filename in filenames])
