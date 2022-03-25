import cv2

import person_detector

def main():
    camera_width  = 640
    camera_height = 640
    vidfps = 30

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, vidfps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    #cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, color_image = cam.read()
        if not ret:
            continue
       
        person_detector.detect(color_image, "./models/yolox_s.pth", "./output/")
        
        #cv2.imwrite("./test.jpg", color_image)
        #cv2.imshow('USB Camera', color_image)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

main()
