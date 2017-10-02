# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from model import DAE

class ModelTest(tf.test.TestCase):
  def test_dae(self):
    dae = DAE()
    
    #vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dae")
    #print(vars)

if __name__ == "__main__":
  tf.test.main()
  
