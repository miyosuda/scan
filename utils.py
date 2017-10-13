# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from scipy.misc import toimage


def convert_hsv_to_rgb(image):
  """ Convert hsv image(0.0~1.0) to rgb image(0.0~1.0) """
  
  scale_hsv = np.array([180.0, 255.0, 255.0], dtype=np.float32)
  bgr = cv2.cvtColor(np.uint8(image * scale_hsv), cv2.COLOR_HSV2BGR) # Convert to RGB
  rgb = bgr[:, :, ::-1] # np.uint8
  scale_rgb = np.array([1.0/255.0, 1.0/255, 1.0/255.0], dtype=np.float32)
  return rgb * scale_rgb


def save_image(image, file_name):
  """ Save RGB image (0.0~1.0) """
  toimage(image, cmin=0, cmax=1.0).save(file_name)
