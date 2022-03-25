# person-detect

object detection based on [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

# Build environment

this program is only support yolox_s 

1. clone and make models directory
```
git clone https://github.com/Ruketa/person-detect.git
cd person-ditect
mkdir models
```

2. download the weights of yolo_s from the below link and copy to models directory

[weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)

3. build container image
```
docker build -t <image-name> .
```

# RUN object detection

- I check this program only on ubuntu 18.04.5 LTS

- Before run object detection, Confirm that the camera is recognized

1. run container
```
docker run --device /dev/video0:/dev/video0 -v $PWD/output:/WORK/output <container-name> 
```