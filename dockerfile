FROM python:3.9.7

WORKDIR /app

COPY ./main.py ./
COPY ./requirements.txt ./
COPY ./src ./src
# COPY ./src/*.py ./src
# COPY ./src/character_recognition/*.py ./src/character_recognition
# COPY ./src/lp_detection/*.py ./src/lp_detection
# COPY ./src/weight_folder/*.h5 ./src/weight_folder
# COPY ./src/weight_folder/*.json ./src/weight_folder
# COPY ./src/weight_folder/config/*.py ./src/weight_folder/config
# COPY ./src/weight_folder/config/*.yml ./src/weight_folder/config
# COPY ./src/weight_folder/model/all_plates/*.pth  ./src/weight_folder/model/all_plates
# COPY ./src/weight_folder/model/all_plates_cut_and_padding/*.pth  ./src/weight_folder/model/all_plates_cut_and_padding
# COPY ./src/weight_folder/model/all_plates_padding/*.pth  ./src/weight_folder/model/all_plates_padding
# COPY ./src/weight_folder/model/pretrain/*.pth ./src/weight_folder/model/pretrain
# COPY ./vietocr/*.py ./vietocr
# COPY ./vietocr/model/*.py ./vietocr/model
# COPY ./vietocr/model/backbone/*.py ./vietocr/model/backbone
# COPY ./vietocr/model/seqmodel/*.py ./vietocr/model/seqmodel
# COPY ./vietocr/tool/*.py ./vietocr/tool
COPY ./vietocr ./vietocr
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python"]
CMD ["main.py"]