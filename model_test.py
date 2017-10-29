# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from model import DAE, VAE, SCAN, SCANRecombinator

class ModelTest(tf.test.TestCase):
  def test_dae(self):
    dae = DAE()
    
    vars = dae.get_vars()
    # Check size of optimizing vars
    self.assertEqual(len(vars), 10+10)

  def test_vae(self):
    dae = DAE()
    vae = VAE(dae)

    vars = vae.get_vars()
    # Check size of optimizing vars
    self.assertEqual(len(vars), 14+12)

  def test_scan(self):
    dae = DAE()
    vae = VAE(dae)
    scan = SCAN(dae, vae)

    vars = scan.get_vars()
    # Check size of optimizing vars
    self.assertEqual(len(vars), 6+4)

  def test_scan_recombinator(self):
    dae = DAE()
    vae = VAE(dae)
    scan = SCAN(dae, vae)
    scan_recomb = SCANRecombinator(dae, vae, scan)

    vars = scan_recomb.get_vars()
    # Check size of optimizing vars
    self.assertEqual(len(vars), 4)


if __name__ == "__main__":
  tf.test.main()
