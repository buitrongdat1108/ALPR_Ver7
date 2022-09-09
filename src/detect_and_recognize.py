import base64
import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image
LP_DETECTION_CFG = {
    "wpod-net_update1.json": "./src/weight_folder/wpod-net_update1.json",
    "wpod-net_update1.h5": "./src/weight_folder/wpod-net_update1.h5"
}

def load_wpod_net_model():
    try:
        #path = splitext(path)[0]
        with open(LP_DETECTION_CFG["wpod-net_update1.json"], 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights(LP_DETECTION_CFG["wpod-net_update1.h5"])
        print("[INFO] Wpod-Net_update1.h5 loaded successfully...")
        return model
    except Exception as e:
        print(e)

#convert base64-type to raw image
def cvtBase64(base64Img):
    try:
        base64Img = np.fromstring(base64.b64decode(base64Img), dtype=np.uint8)
        base64Img = cv2.imdecode(base64Img, cv2.IMREAD_COLOR)
    except:
        return None
    return base64Img

def paddingImg(inputImg, resize=False):
    padding = np.zeros([inputImg.shape[1], inputImg.shape[1], 3])
    a = int((inputImg.shape[1] - inputImg.shape[0]) / 2)
    padding[a:(inputImg.shape[0] + a), 0:inputImg.shape[1], :] = inputImg
    inputImg = cv2.resize(inputImg, (480, 480))
    inputImg = inputImg / 255
    if resize:
        inputImg = cv2.resize(inputImg, (720, 720))
    return inputImg

def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts

# image_count = 47176

def get_plate_numbers_for_rec_plate(plate_image):
    # global image_count
    # image_count = image_count + 1
    padding = np.zeros([470, 470, 3])
    padding[180:290, 0:470, :] = plate_image
    # padding = cv2.cvtColor(np.float32(padding), cv2.COLOR_BGR2GRAY)
    # padding = cv2.merge((padding, padding, padding))
    # cv2.imwrite("temp.jpg", padding)
    padding_resize = Image.fromarray(np.uint8(padding)).convert('RGB')
    # path = "/home/vdtc/ALPR/test_components/data_2022_03_29/plate_image_train/%d.jpg" % image_count
    # padding_resize.save(path)
    return padding_resize


def get_plate_numbers_for_square_plate(plate_image):
    # global image_count
    # image_count = image_count + 1
    padding = np.zeros([470, 470, 3])
    padding[135:335, 95:375, :] = plate_image
    padding_resize = Image.fromarray(np.uint8(padding)).convert('RGB')
    # path = "/home/vdtc/ALPR/test_components/data_2022_03_29/plate_image_train/%d.jpg" % image_count
    # padding_resize.save(path)
    return padding_resize


