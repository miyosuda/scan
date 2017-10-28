# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf


def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


class ModelBase(object):
  def __init__(self):
    """ Auto Encoder base class. """
    pass

  def _conv2d_weight_variable(self, weight_shape, name, deconv=False):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
      input_channels  = weight_shape[3]
      output_channels = weight_shape[2]
    else:
      input_channels  = weight_shape[2]
      output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
    bias   = tf.get_variable(name_b, bias_shape,
                             initializer=conv_initializer(w, h, input_channels))
    return weight, bias


  def _conv1d_weight_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
      
    w = weight_shape[0]
    input_channels  = weight_shape[1]
    output_channels = weight_shape[2]
    d = 1.0 / np.sqrt(input_channels * w)
    bias_shape = [output_channels]
  
    weight = tf.get_variable(name_w, weight_shape,
                             initializer=fc_initializer(input_channels))
                             #initializer=conv_initializer(w, 1, input_channels))
    bias   = tf.get_variable(name_b, bias_shape,
                             initializer=fc_initializer(input_channels))
                             #initializer=conv_initializer(w, 1, input_channels))
    return weight, bias


  def _fc_weight_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
    return weight, bias
  
  
  def _get_deconv2d_output_size(self, input_height, input_width, filter_height,
                                filter_width, row_stride, col_stride, padding_type):
    if padding_type == 'VALID':
      out_height = (input_height - 1) * row_stride + filter_height
      out_width  = (input_width  - 1) * col_stride + filter_width
    elif padding_type == 'SAME':
      out_height = input_height * row_stride
      out_width  = input_width * col_stride
    return out_height, out_width
  
  
  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                        padding='SAME')
  
  
  def _deconv2d(self, x, W, input_width, input_height, stride):
    filter_height = W.get_shape()[0].value
    filter_width  = W.get_shape()[1].value
    out_channel   = W.get_shape()[2].value
    
    out_height, out_width = self._get_deconv2d_output_size(input_height,
                                                           input_width,
                                                           filter_height,
                                                           filter_width,
                                                           stride,
                                                           stride,
                                                           'SAME')
    batch_size = tf.shape(x)[0]
    output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')

  def _sample_z(self, z_mean, z_log_sigma_sq):
    eps_shape = tf.shape(z_mean)
    eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
    # z = mu + sigma * epsilon
    z = tf.add(z_mean,
               tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    return z

  
  def _kl(self, mu1, log_sigma1_sq, mu2, log_sigma2_sq):
    return tf.reduce_sum(0.5 * (log_sigma2_sq - log_sigma1_sq +
                                tf.exp(log_sigma1_sq - log_sigma2_sq) +
                                tf.square(mu1 - mu2) / tf.exp(log_sigma2_sq) -
                                1))

  

class DAE(ModelBase):
  """ Denoising Auto Encoder. """
  
  def __init__(self, learning_rate=1e-4, epsilon=1e-8):
    ModelBase.__init__(self)
    
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    
    # Create autoencoder network
    self._create_network()
    
    # Define loss function and corresponding optimizer
    self._create_loss_optimizer()
      
  
  def _create_recognition_network(self, x, reuse=False):
    with tf.variable_scope("rec", reuse=reuse) as scope:
      # [filter_height, filter_width, in_channels, out_channels]
      W_conv1, b_conv1 = self._conv2d_weight_variable([4, 4, 3,  32], "conv1")
      W_conv2, b_conv2 = self._conv2d_weight_variable([4, 4, 32, 32], "conv2")
      W_conv3, b_conv3 = self._conv2d_weight_variable([4, 4, 32, 64], "conv3")
      W_conv4, b_conv4 = self._conv2d_weight_variable([4, 4, 64, 64], "conv4")
      W_fc1, b_fc1     = self._fc_weight_variable([5 * 5 * 64, 100], "fc1")

      h_conv1 = tf.nn.elu(self._conv2d(x, W_conv1, 2) + b_conv1)        # (40, 40)
      h_conv2 = tf.nn.elu(self._conv2d(h_conv1, W_conv2, 2) + b_conv2)  # (20, 20)
      h_conv3 = tf.nn.elu(self._conv2d(h_conv2, W_conv3, 2) + b_conv3)  # (10, 10)
      h_conv4 = tf.nn.elu(self._conv2d(h_conv3, W_conv4, 2) + b_conv4)  # (5, 5)
      h_conv4_flat = tf.reshape(h_conv4, [-1, 5 * 5 * 64])
      z = tf.tanh(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
      return z

  
  def _create_generator_network(self, z, reuse=False):
    with tf.variable_scope("gen", reuse=reuse) as scope:
      W_fc1, b_fc1 = self._fc_weight_variable([100, 5 * 5 * 64], "fc1")
      
      # [filter_height, filter_width, output_channels, in_channels]
      W_deconv1, b_deconv1 = self._conv2d_weight_variable([4, 4, 64, 64], "deconv1",
                                                        deconv=True)
      W_deconv2, b_deconv2 = self._conv2d_weight_variable([4, 4, 32, 64], "deconv2",
                                                        deconv=True)
      W_deconv3, b_deconv3 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv3",
                                                        deconv=True)
      W_deconv4, b_deconv4 = self._conv2d_weight_variable([4, 4, 3, 32],  "deconv4",
                                                        deconv=True)

      h_fc1 = tf.nn.elu(tf.matmul(z, W_fc1) + b_fc1)
      h_fc1_reshaped = tf.reshape(h_fc1, [-1, 5, 5, 64])
      h_deconv1 = tf.nn.elu(self._deconv2d(h_fc1_reshaped, W_deconv1, 5, 5, 2) + b_deconv1)
      h_deconv2 = tf.nn.elu(self._deconv2d(h_deconv1, W_deconv2, 10, 10, 2) + b_deconv2)
      h_deconv3 = tf.nn.elu(self._deconv2d(h_deconv2, W_deconv3, 20, 20, 2) + b_deconv3)
      x_out =    tf.sigmoid(self._deconv2d(h_deconv3, W_deconv4, 40, 40, 2) + b_deconv4)
      return x_out


  def _create_network(self):
    # tf Graph input 
    self.x     = tf.placeholder(tf.float32, shape=[None, 80, 80, 3]) # Masked image input
    self.x_org = tf.placeholder(tf.float32, shape=[None, 80, 80, 3]) # Original image input
    
    with tf.variable_scope("dae"):
      self.z     = self._create_recognition_network(self.x)
      self.x_out = self._create_generator_network(self.z)
      
    
  def _create_loss_optimizer(self):
    # Reconstruction loss
    reconstr_loss = 0.5 * tf.reduce_sum( tf.square(self.x_org - self.x_out) )
    self.loss = reconstr_loss

    loss_summary_op = tf.summary.scalar('dae_loss', reconstr_loss)
    self.summary_op = tf.summary.merge([loss_summary_op])

    self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dae")
    
    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate,
      epsilon=self.epsilon).minimize(self.loss, var_list=self.variables)

    
  def partial_fit(self, sess, xs_masked, xs_org, summary_writer, step):
    """Train model based on mini-batch of input data.
    
    Return loss of mini-batch.
    """
    _, loss, summary_str = sess.run((self.optimizer, self.loss, self.summary_op), 
                                    feed_dict={self.x: xs_masked,
                                               self.x_org: xs_org})
    summary_writer.add_summary(summary_str, step)
    return loss


  def reconstruct(self, sess, xs):
    """ Reconstruct given data. """
    return sess.run(self.x_out, 
                    feed_dict={self.x: xs})

  
  def get_vars(self):
    return self.variables


class VAE(ModelBase):
  """ Beta Variational Auto Encoder. """
  
  def __init__(self, dae, beta=53.0, learning_rate=1e-4, epsilon=1e-8):
    ModelBase.__init__(self)

    self.beta = beta
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    
    # Create autoencoder network
    self._create_network(dae)
    
    # Define loss function and corresponding optimizer
    self._create_loss_optimizer()
    
  
  def _create_recognition_network(self, x, reuse=False):
    with tf.variable_scope("rec", reuse=reuse) as scope:
      # [filter_height, filter_width, in_channels, out_channels]
      W_conv1, b_conv1 = self._conv2d_weight_variable([4, 4, 3,  32], "conv1")
      W_conv2, b_conv2 = self._conv2d_weight_variable([4, 4, 32, 32], "conv2")
      W_conv3, b_conv3 = self._conv2d_weight_variable([4, 4, 32, 64], "conv3")
      W_conv4, b_conv4 = self._conv2d_weight_variable([4, 4, 64, 64], "conv4")
      W_fc1, b_fc1     = self._fc_weight_variable([5 * 5 * 64, 256], "fc1")
      W_fc2, b_fc2     = self._fc_weight_variable([256, 32], "fc2")
      W_fc3, b_fc3     = self._fc_weight_variable([256, 32], "fc3")
      
      h_conv1 = tf.nn.relu(self._conv2d(x, W_conv1, 2) + b_conv1)        # (40, 40)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2, 2) + b_conv2)  # (20, 20)
      h_conv3 = tf.nn.relu(self._conv2d(h_conv2, W_conv3, 2) + b_conv3)  # (10, 10)
      h_conv4 = tf.nn.relu(self._conv2d(h_conv3, W_conv4, 2) + b_conv4)  # (5, 5)
      h_conv4_flat = tf.reshape(h_conv4, [-1, 5 * 5 * 64])
      h_fc = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
    
      z_mean         = tf.matmul(h_fc, W_fc2) + b_fc2
      z_log_sigma_sq = tf.matmul(h_fc, W_fc3) + b_fc3
      return (z_mean, z_log_sigma_sq)

  
  def _create_generator_network(self, z, reuse=False):
    with tf.variable_scope("gen", reuse=reuse) as scope:    
      W_fc1, b_fc1 = self._fc_weight_variable([32, 256], "fc1")
      W_fc2, b_fc2 = self._fc_weight_variable([256, 5 * 5 * 64], "fc2")

      # [filter_height, filter_width, output_channels, in_channels]
      W_deconv1, b_deconv1 = self._conv2d_weight_variable([4, 4, 64, 64], "deconv1", deconv=True)
      W_deconv2, b_deconv2 = self._conv2d_weight_variable([4, 4, 32, 64], "deconv2", deconv=True)
      W_deconv3, b_deconv3 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv3", deconv=True)
      W_deconv4, b_deconv4 = self._conv2d_weight_variable([4, 4,  3, 32], "deconv4", deconv=True)

      h_fc1 = tf.nn.relu(tf.matmul(z,     W_fc1) + b_fc1)
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
      h_fc2_reshaped = tf.reshape(h_fc2, [-1, 5, 5, 64])
      h_deconv1 = tf.nn.relu(self._deconv2d(h_fc2_reshaped, W_deconv1, 5,   5, 2) + b_deconv1)
      h_deconv2 = tf.nn.relu(self._deconv2d(h_deconv1,      W_deconv2, 10, 10, 2) + b_deconv2)
      h_deconv3 = tf.nn.relu(self._deconv2d(h_deconv2,      W_deconv3, 20, 20, 2) + b_deconv3)
      x_out = tf.sigmoid(self._deconv2d(h_deconv3, W_deconv4, 40, 40, 2) + b_deconv4)
      return x_out

    
  def _create_network(self, dae):
    # tf Graph input
    self.x = tf.placeholder(tf.float32, shape=[None, 80, 80, 3])
    
    with tf.variable_scope("vae"):
      self.z_mean, self.z_log_sigma_sq = self._create_recognition_network(self.x)

      # Draw one sample z from Gaussian distribution
      # z = mu + sigma * epsilon
      self.z = self._sample_z(self.z_mean, self.z_log_sigma_sq)
      self.x_out = self._create_generator_network(self.z)

    with tf.variable_scope("dae", reuse=True):
      self.z_d     = dae._create_recognition_network(self.x,     reuse=True)
      self.z_out_d = dae._create_recognition_network(self.x_out, reuse=True)
      self.x_d     = dae._create_generator_network(self.z_d,     reuse=True)
      self.x_out_d = dae._create_generator_network(self.z_out_d, reuse=True)
      
      
  def _create_loss_optimizer(self):
    # Reconstruction loss
    self.reconstr_loss = 0.5 * tf.reduce_sum( tf.square(self.z_d - self.z_out_d) )

    reconstr_loss_summary_op = tf.summary.scalar('vae_reconstr_loss', self.reconstr_loss)

    # Latent loss
    self.latent_loss = self.beta * -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                                        - tf.square(self.z_mean) 
                                                        - tf.exp(self.z_log_sigma_sq))

    latent_loss_summary_op = tf.summary.scalar('vae_latent_loss', self.latent_loss)

    self.summary_op = tf.summary.merge([reconstr_loss_summary_op, latent_loss_summary_op])
    
    self.loss = self.reconstr_loss + self.latent_loss

    # DAE part is not trained.
    self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vae")
    
    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate,
      epsilon=self.epsilon).minimize(self.loss, var_list=self.variables)

    
  def partial_fit(self, sess, xs, summary_writer, step):
    """Train model based on mini-batch of input data.
    
    Return loss of mini-batch.
    """
    _, reconstr_loss, latent_loss, summary_str = sess.run((self.optimizer,
                                                           self.reconstr_loss,
                                                           self.latent_loss,
                                                           self.summary_op),
                                                          feed_dict={self.x: xs})
    summary_writer.add_summary(summary_str, step)
    return reconstr_loss, latent_loss


  def reconstruct(self, sess, xs, through_dae=True):
    """ Reconstruct given data. """
    if through_dae:
      # Use output from DAE decoder
      return sess.run(self.x_out_d, 
                      feed_dict={self.x: xs})
    else:
      # Original VAE output
      return sess.run(self.x_out, 
                      feed_dict={self.x: xs})

  
  def transform(self, sess, xs):
    """Transform data by mapping it into the latent space."""
    return sess.run( [self.z_mean, self.z_log_sigma_sq],
                     feed_dict={self.x: xs} )
  

  def generate(self, sess, zs):
    """ Generate data by sampling from latent space. """
    return sess.run( self.x_out_d, 
                     feed_dict={self.z: zs} )
  
  
  def get_vars(self):
    return self.variables



class SCAN(ModelBase):
  """ SCAN Auto Encoder. """

  def __init__(self, dae, vae, beta=1.0, lambd=10.0, learning_rate=1e-4, epsilon=1e-8):
    ModelBase.__init__(self)
    
    self.beta = beta
    self.lambd = lambd
    self.learning_rate = learning_rate
    self.epsilon = epsilon

    # Create autoencoder network
    self._create_network(dae, vae)

    # Define loss function and corresponding optimizer
    self._create_loss_optimizer()

    
  def _create_recognition_network(self, y, reuse=False):
    with tf.variable_scope("rec", reuse=reuse) as scope:
      # [filter_height, filter_width, in_channels, out_channels]
      W_fc1, b_fc1     = self._fc_weight_variable([51, 100], "fc1")
      W_fc2, b_fc2     = self._fc_weight_variable([100, 32], "fc2")
      W_fc3, b_fc3     = self._fc_weight_variable([100, 32], "fc3")
      
      h_fc = tf.nn.relu(tf.matmul(y, W_fc1) + b_fc1)
      z_mean         = tf.matmul(h_fc, W_fc2) + b_fc2
      z_log_sigma_sq = tf.matmul(h_fc, W_fc3) + b_fc3
      return (z_mean, z_log_sigma_sq)

    
  def _create_generator_network(self, z, reuse=False):
    with tf.variable_scope("gen", reuse=reuse) as scope:
      W_fc1, b_fc1 = self._fc_weight_variable([32, 100], "fc1")
      W_fc2, b_fc2 = self._fc_weight_variable([100, 51], "fc2")

      h_fc1 = tf.nn.relu(tf.matmul(z,     W_fc1) + b_fc1)
      y_out_logit = tf.matmul(h_fc1, W_fc2) + b_fc2
      y_out = tf.sigmoid(y_out_logit)
      return y_out_logit, y_out

    
  def _create_network(self, dae, vae):
    # tf Graph input
    self.x = tf.placeholder(tf.float32, shape=[None, 80, 80, 3])
    self.y = tf.placeholder(tf.float32, shape=[None, 51])

    # Create SCAN training network
    with tf.variable_scope("scan"):
      self.z_mean, self.z_log_sigma_sq = self._create_recognition_network(self.y)
      self.z = self._sample_z(self.z_mean, self.z_log_sigma_sq)
      self.y_out_logit, self.y_out = self._create_generator_network(self.z)

    with tf.variable_scope("vae", reuse=True):
      self.x_z_mean, self.x_z_log_sigma_sq = vae._create_recognition_network(self.x,
                                                                             reuse=True)
      self.x_z = self._sample_z(self.x_z_mean, self.x_z_log_sigma_sq)
      self.x_out = vae._create_generator_network(self.x_z, reuse=True)

    with tf.variable_scope("dae", reuse=True):
      self.z_out_d = dae._create_recognition_network(self.x_out, reuse=True)
      self.x_out_d = dae._create_generator_network(self.z_out_d, reuse=True)

    # Create sym2img network
    with tf.variable_scope("vae", reuse=True):
      x_s2i = vae._create_generator_network(self.z, reuse=True)

    with tf.variable_scope("dae", reuse=True):
      z_d_s2i = dae._create_recognition_network(x_s2i, reuse=True)
      self.x_d_s2i = dae._create_generator_network(z_d_s2i, reuse=True)

    # Create img2sym network
    with tf.variable_scope("scan", reuse=True):
      _, self.y_i2s = self._create_generator_network(self.x_z, reuse=True)


      
  def _create_loss_optimizer(self):
    # Reconstruction loss
    self.reconstr_loss = tf.reduce_sum(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.y_out_logit))

    reconstr_loss_summary_op = tf.summary.scalar('scan_reconstr_loss', self.reconstr_loss)

    # Latent loss
    self.latent_loss0 = self.beta * -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                                         - tf.square(self.z_mean)
                                                         - tf.exp(self.z_log_sigma_sq))
    self.latent_loss1 = self.lambd * self._kl(self.x_z_mean, self.x_z_log_sigma_sq,
                                              self.z_mean, self.z_log_sigma_sq)

    latent_loss0_summary_op = tf.summary.scalar('scan_latent_loss0', self.latent_loss0)
    latent_loss1_summary_op = tf.summary.scalar('scan_latent_loss1', self.latent_loss1)

    self.summary_op = tf.summary.merge([reconstr_loss_summary_op,
                                        latent_loss0_summary_op,
                                        latent_loss1_summary_op])

    self.loss = self.reconstr_loss + self.latent_loss0 + self.latent_loss1
    
    self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="scan")
    
    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate,
      epsilon=self.epsilon).minimize(self.loss, var_list=self.variables)

    
  def partial_fit(self, sess, xs, ys, summary_writer, step):
    """Train model based on mini-batch of input data.
    
    Return loss of mini-batch.
    """
    _, reconstr_loss, latent_loss0, latent_loss1, summary_str = sess.run((self.optimizer,
                                                                          self.reconstr_loss,
                                                                          self.latent_loss0,
                                                                          self.latent_loss1,
                                                                          self.summary_op),
                                                                         feed_dict={self.x: xs,
                                                                                    self.y: ys})
    summary_writer.add_summary(summary_str, step)
    return reconstr_loss, latent_loss0, latent_loss1


  def generate_from_labels(self, sess, ys):
    """ Generate image data from labels. (sym2img) """
    return sess.run( self.x_d_s2i, 
                     feed_dict={self.y: ys} )

  
  def generate_from_images(self, sess, xs):
    """ Generate labels from images. (img2sym) """
    return sess.run( self.y_i2s, 
                     feed_dict={self.x: xs} )

  
  def get_vars(self):
    return self.variables



class SCANRecombinator(ModelBase):
  """ SCAN concept recombinator. """

  def __init__(self, vae, scan, learning_rate=1e-3, epsilon=1e-8):
    ModelBase.__init__(self)

    self.learning_rate = learning_rate
    self.epsilon = epsilon

    # Create network
    self._create_network(vae, scan)

    # Define loss function and corresponding optimizer
    self._create_loss_optimizer()

  def _create_network(self, vae, scan):
    # tf Graph input
    self.y0 = tf.placeholder(tf.float32, shape=[None, 51])
    self.y1 = tf.placeholder(tf.float32, shape=[None, 51])
    self.x = tf.placeholder(tf.float32, shape=[None, 80, 80, 3])
    self.h = tf.placeholder(tf.int32, shape=[None])

    with tf.variable_scope("scan", reuse=True):
      z_mean0, z_log_sigma_sq0 = scan._create_recognition_network(self.y0)
      z_mean1, z_log_sigma_sq1 = scan._create_recognition_network(self.y1)
        
      z_stacked = tf.stack([z_mean0, z_mean1, z_log_sigma_sq0, z_log_sigma_sq1], axis=2)
      # (-1,32,4)

    with tf.variable_scope("scan_recomb"):
      h_onehot = tf.one_hot(indices=self.h, depth = 3)
      # (-1, 3)
      h_onehot = tf.reshape(h_onehot, [-1, 1, 3])

      W_conv1, b_conv1 = self._conv1d_weight_variable([1, 4, 1024], "conv1")
      # (1,4,1024), (1024,)
      W_conv2, b_conv2 = self._conv1d_weight_variable([1, 1024, 6], "conv2")
      # (1,1024,6), (6,)
      
      h_conv1 = tf.nn.relu(tf.nn.conv1d(z_stacked, W_conv1, stride=1, padding='SAME') + b_conv1)
      # (-1,32,32)
      h_conv2 = tf.nn.conv1d(h_conv1,   W_conv2, stride=1, padding='SAME') + b_conv2
      # (-1,32,6)

      z_means, z_log_sigma_sqs = tf.split(h_conv2, num_or_size_splits=2, axis=2)
      # (-1,32,3) (-1,32,3)

      self.r_z_mean          = tf.reduce_sum(tf.multiply(z_means,         h_onehot), 2)
      self.r_z_log_sigma_sq  = tf.reduce_sum(tf.multiply(z_log_sigma_sqs, h_onehot), 2)
      # (-1, 32)
      
      self.r_z = self._sample_z(self.r_z_mean, self.r_z_log_sigma_sq)

    with tf.variable_scope("scan", reuse=True):
      _, self.y_out = scan._create_generator_network(self.r_z)
      
    with tf.variable_scope("vae", reuse=True):
      self.x_z_mean, self.x_z_log_sigma_sq = vae._create_recognition_network(self.x,
                                                                             reuse=True)
      

  def _create_loss_optimizer(self):
    self.loss = self._kl(self.x_z_mean, self.x_z_log_sigma_sq,
                         self.r_z_mean, self.r_z_log_sigma_sq)

    loss_summary_op = tf.summary.scalar('scan_recomb_loss', self.loss)
    self.summary_op = tf.summary.merge([loss_summary_op])

    self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="scan_recomb")
    
    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate,
      epsilon=self.epsilon).minimize(self.loss, var_list=self.variables)


  def partial_fit(self, sess, ys0, ys1, xs, h, summary_writer, step):
    """Train model based on mini-batch of input data.
    
    Return loss of mini-batch.
    """
    _, loss, summary_str  = sess.run((self.optimizer,
                                      self.loss,
                                      self.summary_op),
                                     feed_dict={self.y0: ys0,
                                                self.y1: ys1,
                                                self.h: h,
                                                self.x: xs})
    summary_writer.add_summary(summary_str, step)
    return loss

  
  def recombinate(self, sess, ys0, ys1, hs):
    """ Recominate labels. """
    return sess.run( self.y_out,
                     feed_dict={self.y0: ys0,
                                self.y1: ys1,
                                self.h: hs} )
  
  
  def get_vars(self):
    return self.variables
