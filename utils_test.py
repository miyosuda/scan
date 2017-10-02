# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from utils import *


class UtilsTest(unittest.TestCase):
  def test_convert_hsv_to_rgb(self):
    h = 1.0/6.0 # Yello
    s = 1.0
    v = 1.0
    hsv = np.float32([[[h, s, v]]])

    rgb = convert_hsv_to_rgb(hsv)
    
    self.assertTrue( rgb.shape == (1,1,3) )
    self.assertAlmostEqual( rgb[0][0][0], 1.0 )
    self.assertAlmostEqual( rgb[0][0][1], 1.0 )
    self.assertAlmostEqual( rgb[0][0][2], 0.0 )
    

if __name__ == '__main__':
  unittest.main()
