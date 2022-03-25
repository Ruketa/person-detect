FROM python:3.8-buster

WORKDIR /WORK

RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git

RUN apt-get update \
	&& apt-get install -y libgl1-mesa-dev \
	&& pip install -r YOLOX/requirements.txt \
	&& mkdir models

COPY ./models/yolox_s.pth /WORK/models
COPY ./*.py /WORK/

CMD python detect_runner.py
