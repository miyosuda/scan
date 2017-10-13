# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from scipy.misc import toimage
import matplotlib.pyplot as plt
from model import DAE, VAE, SCAN
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
              training_epochs=3000,
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
              training_epochs=3000,
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

      
def train_scan(session,
               scan,
               data_manager,
               saver,
               summary_writer,
               batch_size=16,
               training_epochs=3000,
               display_epoch=1,
               save_epoch=50):

  step = 0
  
  for epoch in range(training_epochs):
    average_reconstr_loss = 0.0
    average_latent_loss0  = 0.0
    average_latent_loss1  = 0.0
    total_batch = int(n_samples / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # Get batch of images
      batch_xs, batch_ys = data_manager.next_batch(batch_size, use_labels=True)
      
      # Fit training using batch data
      reconstr_loss, latent_loss0, latent_loss1 = scan.partial_fit(session, batch_xs, batch_ys,
                                                                   summary_writer, step)
      
      # Compute average loss
      average_reconstr_loss += reconstr_loss / n_samples * batch_size
      average_latent_loss0  += latent_loss0  / n_samples * batch_size
      average_latent_loss1  += latent_loss1  / n_samples * batch_size

      step += 1
      
     # Display logs per epoch step
    if epoch % display_epoch == 0:
      print("Epoch:", '%04d' % (epoch+1),
            "reconstr=", "{:.3f}".format(average_reconstr_loss),
            "latent0=",  "{:.3f}".format(average_latent_loss0),
            "latent1=",  "{:.3f}".format(average_latent_loss1))

    """
    if epoch % 10 == 0:
      reconstruct_xs = scan.reconstruct(session, batch_xs)
      hsv_image = reconstruct_xs[0].reshape((80,80,3))
      rgb_image = utils.convert_hsv_to_rgb(hsv_image)
      plt.figure()
      plt.imshow(rgb_image)
      plt.savefig('reconstr.png')
      plt.close()
    """

    # Save to checkpoint
    if epoch % save_epoch == 0:
      saver.save(session, epoch)


def disentangle_check(session, vae, data_manager, save_original=True):
  hsv_image = data_manager.get_image(obj_color=0, wall_color=0, floor_color=0, obj_id=0)
  rgb_image = utils.convert_hsv_to_rgb(hsv_image)
  if save_original:
    toimage(rgb_image, cmin=0, cmax=1.0).save("original.png")

  batch_xs = [hsv_image]
  z_mean, z_log_sigma_sq = vae.transform(session, batch_xs)
  z_sigma_sq = np.exp(z_log_sigma_sq)[0]

  # Print variance
  zss_str = ""
  for i,zss in enumerate(z_sigma_sq):
    str = "z{0}={1:.2f}".format(i,zss)
    zss_str += str + ", "
  print(zss_str)

  # Save disentangled images
  z_m = z_mean[0]
  n_z = 32

  if not os.path.exists("disentangle_img"):
    os.mkdir("disentangle_img")

  for target_z_index in range(n_z):
    for ri in range(10):
      value = -3.0 + (6.0 / 9.0) * ri
      z_mean2 = np.zeros((1, n_z))
      for i in range(n_z):
        if( i == target_z_index ):
          z_mean2[0][i] = value
        else:
          z_mean2[0][i] = z_m[i]
      generated_xs = vae.generate(session, z_mean2)
      hsv_reconstr_img = generated_xs.reshape((80,80,3))
      rgb_reconstr_img = utils.convert_hsv_to_rgb(hsv_reconstr_img)
      toimage(rgb_reconstr_img, cmin=0, cmax=1.0).save(
        "disentangle_img/check_z{0}_{1}.png".format(target_z_index,ri))


def main(argv):
  data_manager = DataManager()
  data_manager.prepare()

  dae = DAE()
  vae = VAE(dae)
  scan = SCAN(dae, vae)

  dae_saver  = CheckPointSaver(CHECKPOINT_DIR, "dae",  dae.get_vars())
  vae_saver  = CheckPointSaver(CHECKPOINT_DIR, "vae",  vae.get_vars())
  scan_saver = CheckPointSaver(CHECKPOINT_DIR, "scan", scan.get_vars())

  sess = tf.Session()

  # Initialze variables
  init = tf.global_variables_initializer()
  sess.run(init)

  # For Tensorboard log
  summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

  # Load from checkpoint
  dae_saver.load(sess)
  vae_saver.load(sess)
  scan_saver.load(sess)

  # Train
  #..train_dae(sess, dae, data_manager, dae_saver, summary_writer)
  #..train_vae(sess, vae, data_manager, vae_saver, summary_writer)

  #..disentangle_check(sess, vae, data_manager)

  train_scan(sess, scan, data_manager, scan_saver, summary_writer)

  sess.close()
  

if __name__ == '__main__':
  tf.app.run()
