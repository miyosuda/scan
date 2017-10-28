# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from scipy.misc import toimage
import matplotlib.pyplot as plt
from model import DAE, VAE, SCAN, SCANRecombinator
import utils
from data_manager import DataManager
from data_manager import IMAGE_CAPACITY, OP_AND, OP_IN_COMMON, OP_IGNORE
from options import get_options


flags = get_options()


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

  print("start training DAE")

  step = 0
  
  for epoch in range(training_epochs):
    average_loss = 0.0
    total_batch = int(IMAGE_CAPACITY / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # Get batch of masked and orignal images
      batch_xs_masked, batch_xs = data_manager.next_masked_batch(batch_size)
      
      # Fit training using batch data
      loss = dae.partial_fit(session, batch_xs_masked, batch_xs,
                             summary_writer, step)
      
      # Compute average loss
      average_loss += loss / IMAGE_CAPACITY * batch_size

      step += 1
      
     # Display logs per epoch step
    if epoch % display_epoch == 0:
      print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.3f}".format(average_loss))

    if epoch % 10 == 0:
      reconstruct_xs = dae.reconstruct(session, batch_xs)
      hsv_image = reconstruct_xs[0].reshape((80,80,3))
      rgb_image = utils.convert_hsv_to_rgb(hsv_image)
      utils.save_image(rgb_image, "reconstr.png")

    # Save to checkpoint
    if (epoch % save_epoch == 0) or (epoch == training_epochs-1):
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

  print("start training Beta-VAE")

  step = 0
  
  for epoch in range(training_epochs):
    average_reconstr_loss = 0.0
    average_latent_loss   = 0.0
    total_batch = int(IMAGE_CAPACITY / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # Get batch of images
      batch_xs = data_manager.next_batch(batch_size)
      
      # Fit training using batch data
      reconstr_loss, latent_loss = vae.partial_fit(session, batch_xs,
                                                   summary_writer, step)
      
      # Compute average loss
      average_reconstr_loss += reconstr_loss / IMAGE_CAPACITY * batch_size
      average_latent_loss   += latent_loss   / IMAGE_CAPACITY * batch_size

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
      utils.save_image(rgb_image, "reconstr.png")
      
    if epoch % 100 == 99:
      disentangle_check(session, vae, data_manager)

    # Save to checkpoint
    if (epoch % save_epoch == 0) or (epoch == training_epochs-1):
      saver.save(session, epoch)


def train_scan(session,
               scan,
               data_manager,
               saver,
               summary_writer,
               batch_size=16,
               training_epochs=1500,
               display_epoch=1,
               save_epoch=50):

  print("start training SCAN")

  step = 0
  
  for epoch in range(training_epochs):
    average_reconstr_loss = 0.0
    average_latent_loss0  = 0.0
    average_latent_loss1  = 0.0
    total_batch = int(IMAGE_CAPACITY / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # Get batch of images
      batch_xs, batch_ys = data_manager.next_batch(batch_size, use_labels=True)
      
      # Fit training using batch data
      reconstr_loss, latent_loss0, latent_loss1 = scan.partial_fit(session, batch_xs, batch_ys,
                                                                   summary_writer, step)
      
      # Compute average loss
      average_reconstr_loss += reconstr_loss / IMAGE_CAPACITY * batch_size
      average_latent_loss0  += latent_loss0  / IMAGE_CAPACITY * batch_size
      average_latent_loss1  += latent_loss1  / IMAGE_CAPACITY * batch_size

      step += 1
      
     # Display logs per epoch step
    if epoch % display_epoch == 0:
      print("Epoch:", '%04d' % (epoch+1),
            "reconstr=", "{:.3f}".format(average_reconstr_loss),
            "latent0=",  "{:.3f}".format(average_latent_loss0),
            "latent1=",  "{:.3f}".format(average_latent_loss1))

    # Save to checkpoint
    if (epoch % save_epoch == 0) or (epoch == training_epochs-1):
      saver.save(session, epoch)

    # Check sym2img and img2sym
    if epoch % 100 == 0:
      sym2img_check(session, scan, data_manager)
      img2sym_check(session, scan, data_manager)

      
def train_scan_recomb(session,
                      scan_recomb,
                      data_manager,
                      saver,
                      summary_writer,
                      batch_size=100,
                      training_epochs=100,
                      display_epoch=1,
                      save_epoch=10):

  print("start training SCAN Recombinator")

  step = 0
  
  for epoch in range(training_epochs):
    average_loss = 0.0
    total_batch = int(IMAGE_CAPACITY / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # Get batch of images
      batch_ys0, batch_ys1, batch_xs, batch_hs = data_manager.get_op_training_batch(batch_size)
      
      # Fit training using batch data
      loss = scan_recomb.partial_fit(session, batch_ys0, batch_ys1, batch_xs, batch_hs,
                                     summary_writer, step)
      
      # Compute average loss
      average_loss += loss / IMAGE_CAPACITY * batch_size

      step += 1
      
     # Display logs per epoch step
    if epoch % display_epoch == 0:
      print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.3f}".format(loss))

    # Save to checkpoint
    if (epoch % save_epoch == 0) or (epoch == training_epochs-1):
      saver.save(session, epoch)


def save_10_images(hsv_images, file_name):
  plt.figure()
  fig, axes = plt.subplots(1, 10, figsize=(10, 1),
                           subplot_kw={'xticks': [], 'yticks': []})
  fig.subplots_adjust(hspace=0.1, wspace=0.1)

  for ax,image in zip(axes.flat, hsv_images):
    hsv_image = image.reshape((80,80,3))
    rgb_image = utils.convert_hsv_to_rgb(hsv_image)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.imshow(rgb_image)

  plt.savefig(file_name, bbox_inches='tight')
  plt.close(fig)
  plt.close()


def disentangle_check(session, vae, data_manager, save_original=False):
  """ Generate disentangled images with Beta VAE """
  hsv_image = data_manager.get_image(obj_color=0, wall_color=0, floor_color=0, obj_id=0)
  rgb_image = utils.convert_hsv_to_rgb(hsv_image)
  
  if save_original:
    utils.save_image(rgb_image, "original.png")

  # Caclulate latent mean and variance of given image.
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
    z_mean2 = np.zeros((10, n_z))
    
    for ri in range(10):
      # Change z mean value from -3.0 to +3.0
      value = -3.0 + (6.0 / 9.0) * ri

      for i in range(n_z):
        if( i == target_z_index ):
          z_mean2[ri][i] = value
        else:
          z_mean2[ri][i] = z_m[i]
    generated_xs = vae.generate(session, z_mean2)
    file_name = "disentangle_img/check_z{0}.png".format(target_z_index)
    save_10_images(generated_xs, file_name)


def sym2img_check_sub(session, scan, y, file_name):
  ys = [y] * 10
  xs = scan.generate_from_labels(session, ys)
  save_10_images(xs, file_name)  

def sym2img_check(session, scan, data_manager):
  """ Check sym2img conversion """
  y0 = data_manager.get_labels(wall_color=0)
  sym2img_check_sub(session, scan, y0, "sym2img0.png")

  y1 = data_manager.get_labels(wall_color=0, floor_color=0)
  sym2img_check_sub(session, scan, y1, "sym2img1.png")

  y2 = data_manager.get_labels(wall_color=0, floor_color=0, obj_color=0)
  sym2img_check_sub(session, scan, y2, "sym2img2.png")

  y3 = data_manager.get_labels(wall_color=0, floor_color=0, obj_color=0, obj_id=0)
  sym2img_check_sub(session, scan, y3, "sym2img3.png")

def img2sym_check_sub(session, scan, data_manager, hsv_image):
  batch_xs = [hsv_image] * 10

  ys = scan.generate_from_images(session, batch_xs)
  for y in ys:
    obj_color, wall_color, floor_color, obj_id = data_manager.choose_labels(y)
    print("obj_color={}, wall_color={}, floor_color={}, obj_id={}".format(obj_color,
                                                                          wall_color,
                                                                          floor_color,
                                                                          obj_id))

def img2sym_check(session, scan, data_manager):
  """ Check img2sym conversion """
  hsv_image0 = data_manager.get_image(obj_color=0, wall_color=0, floor_color=0, obj_id=0)
  print("img2sym: obj_color=0, wall_color=0, floor_color=0, obj_id=0")
  #rgb_image0 = utils.convert_hsv_to_rgb(hsv_image0)
  #utils.save_image(rgb_image0, "img2sym0.png")
  img2sym_check_sub(session, scan, data_manager, hsv_image0)

  hsv_image1 = data_manager.get_image(obj_color=10, wall_color=12, floor_color=5, obj_id=1)
  print("img2sym: obj_color=10, wall_color=12, floor_color=5, obj_id=1")
  #rgb_image1 = utils.convert_hsv_to_rgb(hsv_image1)
  #utils.save_image(rgb_image1, "img2sym1.png")
  img2sym_check_sub(session, scan, data_manager, hsv_image1)


def recombination_check(session, scan_recomb, data_manager):
  # Check OP_AND
  y0 = data_manager.get_labels(obj_color=0)
  y1 = data_manager.get_labels(wall_color=0)
  ys = scan_recomb.recombinate(session, [y0] * 10, [y1] * 10, [OP_AND] * 10)
  print(">> OP_AND (obj_color=0, wall_color=0)")
  for i in range(10):
    obj_color, wall_color, floor_color, obj_id = data_manager.choose_labels(ys[i])
    print("obj_color={}, wall_color={}, floor_color={}, obj_id={}".format(obj_color,
                                                                          wall_color,
                                                                          floor_color,
                                                                          obj_id))

  # Check OP_IN_COMMON
  y0 = data_manager.get_labels(obj_color=0, obj_id=0)
  y1 = data_manager.get_labels(obj_color=0, wall_color=0)
  ys = scan_recomb.recombinate(session, [y0] * 10, [y1] * 10, [OP_IN_COMMON] * 10)
  print(">> OP_IN_COMMON (obj_color=0)")
  for i in range(10):
    obj_color, wall_color, floor_color, obj_id = data_manager.choose_labels(ys[i])
    print("obj_color={}, wall_color={}, floor_color={}, obj_id={}".format(obj_color,
                                                                          wall_color,
                                                                          floor_color,
                                                                          obj_id))

  # Check OP_IGNORE
  y0 = data_manager.get_labels(obj_color=0, floor_color=0)
  y1 = data_manager.get_labels(obj_color=0)
  ys = scan_recomb.recombinate(session, [y0] * 10, [y1] * 10, [OP_IGNORE] * 10)
  print(">> OP_IGNORE (floor_color=0)")
  for i in range(10):      
    obj_color, wall_color, floor_color, obj_id = data_manager.choose_labels(ys[i])
    print("obj_color={}, wall_color={}, floor_color={}, obj_id={}".format(obj_color,
                                                                          wall_color,
                                                                          floor_color,
                                                                          obj_id))

def main(argv):
  data_manager = DataManager()
  data_manager.prepare()

  dae = DAE()
  vae = VAE(dae, beta=flags.vae_beta)
  scan = SCAN(dae, vae, beta=flags.scan_beta, lambd=flags.scan_lambda)
  scan_recomb = SCANRecombinator(vae, scan)

  dae_saver  = CheckPointSaver(flags.checkpoint_dir, "dae",  dae.get_vars())
  vae_saver  = CheckPointSaver(flags.checkpoint_dir, "vae",  vae.get_vars())
  scan_saver = CheckPointSaver(flags.checkpoint_dir, "scan", scan.get_vars())
  scan_recomb_saver = CheckPointSaver(flags.checkpoint_dir, "scan_recomb", scan_recomb.get_vars())

  sess = tf.Session()

  # Initialze variables
  init = tf.global_variables_initializer()
  sess.run(init)

  # For Tensorboard log
  summary_writer = tf.summary.FileWriter(flags.log_file, sess.graph)

  # Load from checkpoint
  dae_saver.load(sess)
  vae_saver.load(sess)
  scan_saver.load(sess)
  scan_recomb_saver.load(sess)

  # Train
  if flags.train_dae:
    train_dae(sess, dae, data_manager, dae_saver, summary_writer)

  if flags.train_vae:
    train_vae(sess, vae, data_manager, vae_saver, summary_writer)

  disentangle_check(sess, vae, data_manager)

  if flags.train_scan:
    train_scan(sess, scan, data_manager, scan_saver, summary_writer)

  sym2img_check(sess, scan, data_manager)
  img2sym_check(sess, scan, data_manager)

  if flags.train_scan_recomb:
    train_scan_recomb(sess, scan_recomb, data_manager, scan_recomb_saver, summary_writer)

  recombination_check(sess, scan_recomb, data_manager)

  sess.close()


if __name__ == '__main__':
  tf.app.run()
