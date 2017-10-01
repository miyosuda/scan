# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from data_manager import DataManager


class DataManagerTest(unittest.TestCase):
  def test_get_masked_image(self):
    image = np.ones((80, 80, 3))
    
    data_manager = DataManager()
    masked_image = data_manager._get_masked_image(image)

    # Shape check
    self.assertTrue( masked_image.shape == (80,80,3) )

    # Every element of masked image should be equal or smaller than original's
    self.assertTrue( (masked_image <= image).all() )
    

  def test_next_masked_batch(self):
    data_manager = DataManager()
    data_manager.prepare()
    
    masked_xs, xs = data_manager.next_masked_batch(100)

    self.assertTrue( len(xs) == 100 )
    self.assertTrue( len(masked_xs) == 100 )

    for i in range(100):
      # Shape check
      self.assertTrue( xs[i].shape == (80,80,3) )
      self.assertTrue( masked_xs[i].shape == (80,80,3) )
      # Every element of masked image should be equal or smaller than original's
      self.assertTrue( (masked_xs[i] <= xs[i]).all() )

    
    

if __name__ == '__main__':
  unittest.main()
