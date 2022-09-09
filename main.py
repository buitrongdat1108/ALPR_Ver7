import flask
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.lp_detection.detect import get_plate
from src.detect_and_recognize import load_wpod_net_model, cvtBase64, paddingImg, get_plate_numbers_for_rec_plate, \
    get_plate_numbers_for_square_plate
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from flask import Flask, request
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
wpod_net = load_wpod_net_model()

config = Cfg.load_config_from_file('./src/weight_folder/config/vgg_transformer.yml')
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False

detector = Predictor(config)
# Build Flask app
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def detect_and_recognize_image(inputImg):
    inputImg = cvtBase64(inputImg)
    inputImg = paddingImg(inputImg)
    plate_image, LpType, cor = get_plate(wpod_net, inputImg, Dmax=700, Dmin=500)
    if LpType == 1:
        output_plate_image = get_plate_numbers_for_rec_plate(plate_image)
        plateString = detector.predict(output_plate_image)
    elif LpType == 2:
        output_plate_image = get_plate_numbers_for_square_plate(plate_image)
        plateString = detector.predict(output_plate_image)
    return plateString


@app.route('/detect_plate_base64', methods=['GET', 'POST'])
@cross_origin(origin='*')
def plateDetect():
    inputImg = request.form.get('image')
    try:
        plate_number = detect_and_recognize_image(inputImg)
        return flask.jsonify(number=plate_number, plate_color="")
    except:
        return flask.jsonify(number="", plate_color="")


if __name__ == '__main__':
    app.run(threaded=True, port=8765, host='0.0.0.0')
