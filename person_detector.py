import cv2 
import sys

import torch

from predictor import Predictor

# YOLOX
sys.path.append("./YOLOX")
from yolox.exp import get_exp

def detect(img, model_path, out_dir):

  # load model
  exp = get_exp(None, "yolox-s")
  model = exp.get_model()
  model.eval()
  # load weights
  ckpt = torch.load(model_path, map_location="cpu")
  model.load_state_dict(ckpt["model"])

  # predict
  test_size=(640, 640)
  predictor = Predictor(model, test_size=test_size)
  outputs = predictor.inference(img)

  # visualize
  height, width = img.shape[:2]
  ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
  img_info = {"id": 0}
  img_info["height"] = height
  img_info["width"] = width
  img_info["raw_img"] = img 
  img_info["ratio"] = ratio
  predictor.visualize(outputs[0], img_info, predictor.confthre, save_dir=out_dir)

def detectByFile(image_name):
  img = cv2.imread(image_name)
  detect(img)
