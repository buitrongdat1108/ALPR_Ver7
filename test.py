import base64
import time
import requests


imgPath = "./test_images/2233785415252553170_1_0_04_20220329011843.jpg"
with open(imgPath, "rb") as image_file:
    encodedByte = base64.b64encode(image_file.read())
    encodedString = encodedByte.decode('utf-8')
t = time.time()
req = requests.post("http://127.0.0.1:8765/detect_plate_base64", data={"image": encodedString})
t = time.time() - t
print(req.text, t)