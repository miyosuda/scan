# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from model import DAE, VAE, SCAN

class ModelTest(tf.test.TestCase):
  def test_dae(self):
    dae = DAE()
    
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dae")
    # Check size of optimizing vars
    self.assertEqual(len(vars), 10+10)

  def test_vae(self):
    dae = DAE()
    vae = VAE(dae)

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vae")
    # Check size of optimizing vars
    self.assertEqual(len(vars), 14+12)

  def test_scan(self):
    dae = DAE()
    vae = VAE(dae)
    scan = SCAN(dae, vae)

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "scan")
    # Check size of optimizing vars
    self.assertEqual(len(vars), 6+4)


if __name__ == "__main__":
  tf.test.main()
