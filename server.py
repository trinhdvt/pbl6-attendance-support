from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
from loader import load_model
import os
import time
import numpy as np
import io
from PIL import Image
from src.utils import image_utils

face_compare, card_detector, card_reader = load_model()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_EXTENSIONS'] = ['jpg', 'jpeg', 'png']
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.getcwd() + "/upload/"


@app.route("/api/test", methods=['GET'])
@cross_origin()
def test():
    return jsonify({'message': 'OK'}), 200


@app.route("/api/amen", methods=['POST'])
@cross_origin()
def run_for_my_life():
    if 'face-img' not in request.files \
            or 'card-img' not in request.files:
        resp = {"message": "Missing parameters"}
        return make_response(jsonify(resp), 400)

    face_img = request.files['face-img']
    card_img = request.files['card-img']
    if not allowed_file([face_img.filename, card_img.filename]):
        resp = {"message": "Not allowed!"}
        return make_response(jsonify(resp), 400)

    face_img_path = app.config['UPLOAD_FOLDER'] + face_img.filename
    card_img_path = app.config['UPLOAD_FOLDER'] + card_img.filename
    face_img.save(face_img_path)
    card_img.save(card_img_path)

    start = time.time()
    is_match, distance = face_compare.predict([face_img_path, card_img_path])
    end = time.time()
    return jsonify({
        "match": np.array(is_match, dtype=bool).tolist(),
        "distance": distance.tolist(),
        "ps-time": end - start
    }), 200


@app.route("/api/card_recognize", methods=['POST'])
def run_for_your_life():
    #
    if 'card-img' not in request.files:
        resp = {"message": "Missing parameters"}
        return make_response(jsonify(resp), 400)

    #
    img_file = request.files['card-img']
    img_bytes = img_file.read()
    img = Image.open(io.BytesIO(img_bytes))

    #
    # results = card_detector.predict(img)
    results = extract_info(img)
    return jsonify(results), 200


def extract_info(card_img):
    extracted_rs = card_detector.detect_crop(card_img, save_crop=False, verbose=True)
    #
    for label, img_list in extracted_rs.items():
        img = image_utils.concat_image(img_list)
        extracted_rs[label] = img

    #
    reader_rs = card_reader.batch_predict(extracted_rs.values(), show_time=True)

    #
    for idx, label in enumerate(extracted_rs.keys()):
        extracted_rs[label] = reader_rs[idx]

    return extracted_rs


def allowed_file(filenames):
    return all(["." in filename and
                filename.rsplit(".", 1)[1].lower() in app.config['UPLOAD_EXTENSIONS']
                for filename in filenames])


if __name__ == '__main__':
    port = 8080
    print(f"Server is listening on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
