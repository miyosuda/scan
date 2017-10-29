# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_options():
  tf.app.flags.DEFINE_boolean("train_dae", True, "whether to train DAE")
  tf.app.flags.DEFINE_boolean("train_vae", True, "whether to train Beta-VAE")
  tf.app.flags.DEFINE_boolean("train_scan", True, "whether to train SCAN")
  tf.app.flags.DEFINE_boolean("train_scan_recomb", True, "whether to train SCAN Recombinator")

  tf.app.flags.DEFINE_float("vae_beta", 0.5, "Beta-VAE beta hyper parameter")
  tf.app.flags.DEFINE_float("scan_beta", 1.0, "SCAN beta hyper parameter")
  tf.app.flags.DEFINE_float("scan_lambda", 10.0, "SCAN lambda hyper parameter")  
  
  tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")
  tf.app.flags.DEFINE_string("log_file", "./log/scan_log", "log file directory")

  return tf.app.flags.FLAGS
