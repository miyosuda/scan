# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave
import matplotlib.pyplot as plt
from model import DAE
from data_manager import DataManager, IMAGE_CAPACITY

CHECKPOINT_DIR = 'checkpoints'

n_samples = IMAGE_CAPACITY

def train_dae(session,
              dae,
              data_manager,
              saver,
              batch_size=100,
              training_epochs=1500,
              display_step=1,
              save_step=50):
  
  for epoch in range(training_epochs):
    average_cost = 0.0
    total_batch = int(n_samples / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # Get batch of masked and orignal images
      batch_xs_masked, batch_xs = data_manager.next_masked_batch(batch_size)
      
      # Fit training using batch data
      cost = dae.partial_fit(sess, batch_xs_masked, batch_xs)
      
      # Compute average loss
      average_cost += cost / n_samples * batch_size
      
     # Display logs per epoch step
    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(average_cost))

    if epoch % 10 == 0:
      reconstruct_xs = dae.reconstruct(sess, batch_xs)
      plt.figure()
      plt.imshow(reconstruct_xs[0].reshape((80,80,3)))
      plt.savefig('reconstr.png')

    # Save to checkpoint
    if epoch % save_step == 0:
      saver.save(session, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = epoch)

    
def load_checkpoints(sess):
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")
    if not os.path.exists(CHECKPOINT_DIR):
      os.mkdir(CHECKPOINT_DIR)
  return saver



data_manager = DataManager()
data_manager.prepare()

dae = DAE()

sess = tf.Session()

# Initialze variables
init = tf.global_variables_initializer()
sess.run(init)

# Load from checkpoint
saver = load_checkpoints(sess)

# Train
train_dae(sess, dae, data_manager, saver)

sess.close()
