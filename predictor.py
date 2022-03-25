import cv2 
import time 
import os
import csv
import datetime
import sys

import torch

# YOLOX
sys.path.append("./YOLOX")
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import postprocess, vis

class Predictor(object):
  def __init__(
   self, model, device="cpu", test_size=(640, 640)
  ):
   self.model = model
   self.device = device
   self.test_size = test_size
   self.preproc = ValTransform()
   self.num_classes = COCO_CLASSES
   self.confthre = 0.25 #exp.test_conf
   self.nmsthre = 0.45 #exp.nmsthre
   self.cls_names = COCO_CLASSES

  def inference(self, img) :

    img, _ = self.preproc(img, None, self.test_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()

    with torch.no_grad():
      t0 = time.time()
      outputs = self.model(img)
      outputs = postprocess(outputs, len(self.num_classes), self.confthre, self.nmsthre, class_agnostic=True)

    return outputs

  def visualize(self, output, img_info, cls_conf=0.35, save_dir="./output/") :
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
      return img

    output = output.cpu()

    bboxes = output[:, 0:4]
    bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    person_count = 0
    for cls_idx in cls :
      idx = int(cls_idx.item())
      if idx == 0 :
        person_count += 1

    vis_img = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_dir + "out.jpg", vis_img)

    with open(save_dir + "out.csv", "w") as f:
      writer = csv.writer(f)
      now = datetime.datetime.now() 
      writer.writerow([now, person_count])
 