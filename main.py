# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave
import matplotlib.pyplot as plt
from auto_encoder import AutoEncoder
from image_pool import ImagePool, IMAGE_CAPACITY

CHECKPOINT_DIR = 'checkpoints'

#n_samples = 10000
n_samples = IMAGE_CAPACITY

def train(session,
          model,
          image_pool,
          saver,
          batch_size=100,
          training_epochs=10,
          display_step=1):
  
  for epoch in range(training_epochs):
    average_cost = 0.0
    total_batch = int(n_samples / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # バッチを取得する.
      batch_xs = image_pool.next_batch(batch_size)
      
      # Fit training using batch data
      cost = model.partial_fit(sess, batch_xs)
      
      # Compute average loss
      average_cost += cost / n_samples * batch_size
      
     # Display logs per epoch step
    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(average_cost))

      reconstruct_xs = model.reconstruct(sess, batch_xs)
      print(reconstruct_xs[0].shape)
      plt.imshow(reconstruct_xs[0].reshape((80,80,3)))
      plt.savefig('reconstr.png')

    # checkpointへの保存
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



image_pool = ImagePool()
image_pool.prepare()

model = AutoEncoder()

sess = tf.Session()

# Checkpointからの復元
saver = load_checkpoints(sess)

# Variablesの初期化
init = tf.initialize_all_variables()
sess.run(init)

# 学習
train(sess, model, image_pool, saver)

sess.close()

