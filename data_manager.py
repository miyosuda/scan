# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import cv2
import numpy as np
import random

IMAGE_CAPACITY = 12288


class DataManager(object):
  def __init__(self):
    pass


  def _load_image(self, index, hsv=True):
    file_name = "data/image{}.png".format(index)
    if hsv:
      # [HSV]
      img = cv2.imread(file_name) # opencv returns [BGR] image
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV
      scale = np.array([1.0/180.0, 1.0/255.0, 1.0/255.0],
                       dtype=np.float32) # Scale to fit 0.0 ~ 1.0
      return hsv * scale
    else:
      # [RGB]
      img = np.array(Image.open(file_name), dtype=np.float32)
      return img * (1.0/255.0)
  
  def prepare(self):
    print("start filling image pool")

    self.images = []

    for i in range(IMAGE_CAPACITY):
      image = self._load_image(i)
      self.images.append(image)
      
    print("finish filling image pool")
    
    self._prepare_indices()

    
  def _prepare_indices(self):
    self.image_indices = list(range(IMAGE_CAPACITY))
    random.shuffle(self.image_indices)
    self.used_image_index = 0

    
  def _get_masked_image(self, image):
    masked_image = image.copy()
    shape = masked_image.shape
    
    h = shape[0]
    w = shape[1]

    h0 = np.random.randint(0,h)
    h1 = np.random.randint(0,h)

    w0 = np.random.randint(0,w)
    w1 = np.random.randint(0,w)

    if h0 <= h1:
      hmin = h0
      hmax = h1+1
    else:
      hmin = h1
      hmax = h0+1

    if w0 <= w1:
      wmin = w0
      wmax = w1+1
    else:
      wmin = w1
      wmax = w0+1

    masked_image[hmin:hmax,wmin:wmax,:] = 0.0
    return masked_image

  def next_masked_batch(self, batch_size):
    batch_org = []
    batch_masked = []
    
    for i in range(batch_size):
      index = self.image_indices[self.used_image_index]
      image = self.images[index]
      batch_org.append(image)
      masked_image = self._get_masked_image(image)
      batch_masked.append(masked_image)
      
      self.used_image_index += 1
      if self.used_image_index >= IMAGE_CAPACITY:
        self._prepare_indices()

    return batch_masked, batch_org
