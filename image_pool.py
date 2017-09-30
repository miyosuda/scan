# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import random

#IMAGE_CAPACITY = 12288
IMAGE_CAPACITY = 100

class ImagePool(object):
  def __init__(self):
    pass
  
  def _load_image(self, index):
    file_name = "data/image{}.png".format(index)
    img = np.array( Image.open(file_name), dtype=np.float32)
    # TODO: 1/255するか？
    return img

  def prepare(self):
    print("start filling image pool")

    self.images = []

    for i in range(IMAGE_CAPACITY):
      image = self._load_image(i)
      self.images.append(image)
    
    random.shuffle(self.images)
    
    print("finish filling image pool")

  def next_batch(self, batch_size):
    # TODO: 既にシャッフルされている. バッチ取り切った時に、再シャッフルする
    batch = []
    for i in range(batch_size):
      image_pos = random.randint(0, IMAGE_CAPACITY-1)
      img = self.images[image_pos]
      batch.append(img)
      #new_img = self._get_image()
      #self.images[image_pos] = new_img
    return batch

