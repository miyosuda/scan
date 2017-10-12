# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave
import matplotlib.pyplot as plt
from model import DAE, VAE
import utils
from data_manager import DataManager, IMAGE_CAPACITY

CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = './log/scan_log'

n_samples = IMAGE_CAPACITY

class CheckPointSaver(object):
  def __init__(self, directory, name, variables):
    self.name = name
    self.saver = tf.train.Saver(variables)
    self.save_dir = directory + '/' + name

    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)

  def load(self, session):
    checkpoint = tf.train.get_checkpoint_state(self.save_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
      self.saver.restore(session, checkpoint.model_checkpoint_path)
      print("{}: loaded checkpoint: {}".format(self.name, checkpoint.model_checkpoint_path))
    else:
      print("{}: checkpoint not found".format(self.name))
      
  def save(self, session, global_step):
    self.saver.save(session, self.save_dir + '/checkpoint', global_step=global_step)


def train_dae(session,
              dae,
              data_manager,
              saver,
              summary_writer,
              batch_size=100,
              training_epochs=1500,
              display_epoch=1,
              save_epoch=50):

  step = 0
  
  for epoch in range(training_epochs):
    average_loss = 0.0
    total_batch = int(n_samples / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # Get batch of masked and orignal images
      batch_xs_masked, batch_xs = data_manager.next_masked_batch(batch_size)
      
      # Fit training using batch data
      loss = dae.partial_fit(session, batch_xs_masked, batch_xs,
                             summary_writer, step)
      
      # Compute average loss
      average_loss += loss / n_samples * batch_size

      step += 1
      
     # Display logs per epoch step
    if epoch % display_epoch == 0:
      print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.3f}".format(average_loss))

    if epoch % 10 == 0:
      reconstruct_xs = dae.reconstruct(session, batch_xs)
      hsv_image = reconstruct_xs[0].reshape((80,80,3))
      rgb_image = utils.convert_hsv_to_rgb(hsv_image)
      plt.figure()
      plt.imshow(rgb_image)
      plt.savefig('reconstr.png')
      plt.close()

    # Save to checkpoint
    if epoch % save_epoch == 0:
      saver.save(session, epoch)
      

def train_vae(session,
              vae,
              data_manager,
              saver,
              summary_writer,
              batch_size=100,
              training_epochs=1500,
              display_epoch=1,
              save_epoch=50):

  step = 0
  
  for epoch in range(training_epochs):
    average_reconstr_loss = 0.0
    average_latent_loss   = 0.0
    total_batch = int(n_samples / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # Get batch of images
      batch_xs = data_manager.next_batch(batch_size)
      
      # Fit training using batch data
      reconstr_loss, latent_loss = vae.partial_fit(session, batch_xs,
                                                   summary_writer, step)
      
      # Compute average loss
      average_reconstr_loss += reconstr_loss / n_samples * batch_size
      average_latent_loss   += latent_loss   / n_samples * batch_size

      step += 1
      
     # Display logs per epoch step
    if epoch % display_epoch == 0:
      print("Epoch:", '%04d' % (epoch+1),
            "reconstr=", "{:.3f}".format(average_reconstr_loss),
            "latent=",   "{:.3f}".format(average_latent_loss))

    if epoch % 10 == 0:
      reconstruct_xs = vae.reconstruct(session, batch_xs)
      hsv_image = reconstruct_xs[0].reshape((80,80,3))
      rgb_image = utils.convert_hsv_to_rgb(hsv_image)
      plt.figure()
      plt.imshow(rgb_image)
      plt.savefig('reconstr.png')
      plt.close()

    # Save to checkpoint
    if epoch % save_epoch == 0:
      saver.save(session, epoch)



def main(argv):
  data_manager = DataManager()
  data_manager.prepare()

  dae = DAE()
  vae = VAE(dae)

  dae_saver = CheckPointSaver(CHECKPOINT_DIR, "dae", dae.get_vars())
  vae_saver = CheckPointSaver(CHECKPOINT_DIR, "vae", vae.get_vars())

  sess = tf.Session()

  # Initialze variables
  init = tf.global_variables_initializer()
  sess.run(init)

  # For Tensorboard log
  summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

  # Load from checkpoint
  dae_saver.load(sess)
  vae_saver.load(sess)

  # Train
  train_dae(sess, dae, data_manager, dae_saver, summary_writer)
  train_vae(sess, vae, data_manager, vae_saver, summary_writer)

  sess.close()
  

if __name__ == '__main__':
  tf.app.run()
