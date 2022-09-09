import base64
import time
import requests


imgPath = "/home/datbt/ALPR/data_2022_03_29/camera_bienso/2233785415252557522_1_0_03_20220329021230.jpg"  
with open(imgPath, "rb") as image_file:
    encodedByte = base64.b64encode(image_file.read())
    encodedString = encodedByte.decode('utf-8')
t = time.time()
req = requests.post("http://127.0.0.1:8765/detect_plate_base64", data={"image": encodedString})
t = time.time() - t
print(req.text, t)