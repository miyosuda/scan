# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from data_manager import DataManager
from data_manager import OP_AND, OP_IN_COMMON, OP_IGNORE


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

      # Elements in image is 0.0 ~ 1.0
      self.assertTrue( np.amax(xs[i]) <= 1.0 )
      self.assertTrue( np.amin(xs[i]) >= 0.0 )
      self.assertTrue( np.amax(masked_xs[i]) <= 1.0 )
      self.assertTrue( np.amin(masked_xs[i]) >= 0.0 )

      self.assertTrue( xs[i].dtype == np.float32 )      
      self.assertTrue( masked_xs[i].dtype == np.float32 )


  def test_next_batch(self):
    data_manager = DataManager()
    data_manager.prepare()

    xs, labels = data_manager.next_batch(100, use_labels=True)
    
    self.assertTrue( len(xs) == 100 )
    self.assertTrue( len(labels) == 100 )

    for i in range(100):
      # Shape check
      self.assertTrue( xs[i].shape == (80,80,3) )
      self.assertTrue( labels[i].shape == (51,) )
      
      # Elements in image is 0.0 ~ 1.0
      self.assertTrue( np.amax(xs[i]) <= 1.0 )
      self.assertTrue( np.amin(xs[i]) >= 0.0 )
      self.assertTrue( xs[i].dtype == np.float32 )

      self.assertTrue( np.amax(labels[i]) <= 1.0 )
      self.assertTrue( np.amin(labels[i]) >= 0.0 )
      self.assertTrue( labels[i].dtype == np.float32 )      
    
  
  def test_index_to_labels(self):
    data_manager = DataManager()
    labels = data_manager._index_to_labels(0)

    # Shape check
    self.assertTrue( labels.shape == (51,) )

    for i in range(51):
      if i == 0 or i == 16 or i == 32 or i == 48:
        labels[i] == 1.0
      else:
        labels[i] == 0.0


  def test_choose_op_triplet(self):
    data_manager = DataManager()

    # Check AND operator
    op = OP_AND
    for _ in range(100):
      param0, param1, param_out = data_manager._choose_op_triplet(op)

      self.assertEqual( len(param0), 4 )
      self.assertEqual( len(param1), 4 )
      self.assertEqual( len(param_out), 4 )
      
      for i in range(4):
        p0 = param0[i]
        p1 = param1[i]
        p_out = param_out[i]
        p = -1
        if p0 != -1:
          p = p0
        if p1 != -1:
          p = p1
        self.assertEqual( p, p_out )

    # Check IN_COMMON operator
    op = OP_IN_COMMON
    for _ in range(100):
      param0, param1, param_out = data_manager._choose_op_triplet(op)

      self.assertEqual( len(param0), 4 )
      self.assertEqual( len(param1), 4 )
      self.assertEqual( len(param_out), 4 )
      
      for i in range(4):
        p0 = param0[i]
        p1 = param1[i]
        p_out = param_out[i]
        if p0 != -1:
          if p0 == p1:
            self.assertEqual( p0, p_out )
          else:
            self.assertNotEqual( p0, p_out )
        
    # Check IGNORE_OP operator
    op = OP_IGNORE
    for _ in range(100):
      param0, param1, param_out = data_manager._choose_op_triplet(op)

      self.assertEqual( len(param0), 4 )
      self.assertEqual( len(param1), 4 )
      self.assertEqual( len(param_out), 4 )
      
      for i in range(4):
        p0 = param0[i]
        p1 = param1[i]
        p_out = param_out[i]
        if p0 != -1:
          if p1 != -1:
            self.assertEqual( p_out, -1 )
          else:
            self.assertEqual( p_out, p0 )
        else:
            self.assertEqual( p_out, -1 )
      

if __name__ == '__main__':
  unittest.main()
