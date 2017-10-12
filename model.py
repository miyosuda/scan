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


class AE(object):
  def __init__(self):
    """ Auto Encoder base class. """
    pass

  def _conv_weight_variable(self, weight_shape, name, deconv=False):
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

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
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
  
  
  def _get2d_deconv_output_size(self, input_height, input_width, filter_height,
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
    
    out_height, out_width = self._get2d_deconv_output_size(input_height,
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

  

class DAE(AE):
  """ Denoising Auto Encoder. """
  
  def __init__(self, learning_rate=1e-4, epsilon=1e-8):
    AE.__init__(self)
    
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    
    # Create autoencoder network
    self._create_network()
    
    # Define loss function and corresponding optimizer
    self._create_loss_optimizer()
      
  
  def _create_recognition_network(self, x, reuse=False):
    with tf.variable_scope("rec", reuse=reuse) as scope:
      # [filter_height, filter_width, in_channels, out_channels]
      W_conv1, b_conv1 = self._conv_weight_variable([4, 4, 3,  32], "conv1")
      W_conv2, b_conv2 = self._conv_weight_variable([4, 4, 32, 32], "conv2")
      W_conv3, b_conv3 = self._conv_weight_variable([4, 4, 32, 64], "conv3")
      W_conv4, b_conv4 = self._conv_weight_variable([4, 4, 64, 64], "conv4")
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
      W_deconv1, b_deconv1 = self._conv_weight_variable([4, 4, 64, 64], "deconv1",
                                                        deconv=True)
      W_deconv2, b_deconv2 = self._conv_weight_variable([4, 4, 32, 64], "deconv2",
                                                        deconv=True)
      W_deconv3, b_deconv3 = self._conv_weight_variable([4, 4, 32, 32], "deconv3",
                                                        deconv=True)
      W_deconv4, b_deconv4 = self._conv_weight_variable([4, 4, 3, 32],  "deconv4",
                                                        deconv=True)

      h_fc1 = tf.nn.elu(tf.matmul(z, W_fc1) + b_fc1)
      h_fc1_reshaped = tf.reshape(h_fc1, [-1, 5, 5, 64])
      h_deconv1 = tf.nn.elu(self._deconv2d(h_fc1_reshaped, W_deconv1, 5, 5, 2) + b_deconv1)
      h_deconv2 = tf.nn.elu(self._deconv2d(h_deconv1, W_deconv2, 10, 10, 2) + b_deconv2)
      h_deconv3 = tf.nn.elu(self._deconv2d(h_deconv2, W_deconv3, 20, 20, 2) + b_deconv3)
      x_out = tf.sigmoid(self._deconv2d(h_deconv3, W_deconv4, 40, 40, 2) + b_deconv4)
      return x_out


  def _create_network(self):
    # tf Graph input 
    self.x     = tf.placeholder("float", shape=[None, 80, 80, 3]) # Masked image input
    self.x_org = tf.placeholder("float", shape=[None, 80, 80, 3]) # Original image input
    
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


  def reconstruct(self, sess, X):
    """ Reconstruct given data. """
    return sess.run(self.x_out, 
                    feed_dict={self.x: X})

  
  def get_vars(self):
    return self.variables


class VAE(AE):
  """ Beta Variational Auto Encoder. """
  
  def __init__(self, dae, beta=53.0, learning_rate=1e-4, epsilon=1e-8):
    AE.__init__(self)

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
      W_conv1, b_conv1 = self._conv_weight_variable([4, 4, 3,  32], "conv1")
      W_conv2, b_conv2 = self._conv_weight_variable([4, 4, 32, 32], "conv2")
      W_conv3, b_conv3 = self._conv_weight_variable([4, 4, 32, 64], "conv3")
      W_conv4, b_conv4 = self._conv_weight_variable([4, 4, 64, 64], "conv4")
      W_fc1, b_fc1     = self._fc_weight_variable([5 * 5 * 64, 256], "fc1")
      W_fc2, b_fc2     = self._fc_weight_variable([256, 32], "fc2")
      W_fc3, b_fc3     = self._fc_weight_variable([256, 32], "fc3")
      
      h_conv1 = tf.nn.relu(self._conv2d(x, W_conv1, 2) + b_conv1)        # (40, 40)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2, 2) + b_conv2)  # (20, 20)
      h_conv3 = tf.nn.relu(self._conv2d(h_conv2, W_conv3, 2) + b_conv3)  # (10, 10)
      h_conv4 = tf.nn.relu(self._conv2d(h_conv3, W_conv4, 2) + b_conv4)  # (5, 5)
      h_conv4_flat = tf.reshape(h_conv4, [-1, 5 * 5 * 64])
      h_fc = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
    
      z_mean         = tf.tanh(tf.matmul(h_fc, W_fc2) + b_fc2)
      z_log_sigma_sq = tf.tanh(tf.matmul(h_fc, W_fc3) + b_fc3)
      return (z_mean, z_log_sigma_sq)

  
  def _create_generator_network(self, z, reuse=False):
    with tf.variable_scope("gen", reuse=reuse) as scope:    
      W_fc1, b_fc1 = self._fc_weight_variable([32, 256], "fc1")
      W_fc2, b_fc2 = self._fc_weight_variable([256, 5 * 5 * 64], "fc2")

      # [filter_height, filter_width, output_channels, in_channels]
      W_deconv1, b_deconv1 = self._conv_weight_variable([4, 4, 64, 64], "deconv1", deconv=True)
      W_deconv2, b_deconv2 = self._conv_weight_variable([4, 4, 32, 64], "deconv2", deconv=True)
      W_deconv3, b_deconv3 = self._conv_weight_variable([4, 4, 32, 32], "deconv3", deconv=True)
      W_deconv4, b_deconv4 = self._conv_weight_variable([4, 4,  3, 32], "deconv4", deconv=True)

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
    self.x = tf.placeholder("float", shape=[None, 80, 80, 3])
    
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


  def reconstruct(self, sess, X, through_dae=True):
    """ Reconstruct given data. """
    if through_dae:
      # Use output from DAE decoder
      return sess.run(self.x_out_d, 
                      feed_dict={self.x: X})
    else:
      # Original VAE output
      return sess.run(self.x_out, 
                      feed_dict={self.x: X})

  
  def get_vars(self):
    return self.variables



class SCAN(AE):
  """ SCAN Auto Encoder. """

  def __init__(self, vae, beta=1.0, lambd=10.0, learning_rate=1e-4, epsilon=1e-8):

    self.beta = beta
    self.lambd = lambd
    self.learning_rate = learning_rate
    self.epsilon = epsilon

    self._create_network(vae)
    self._create_loss_optimizer()

    
  def _create_recognition_network(self, y, reuse=False):
    with tf.variable_scope("rec", reuse=reuse) as scope:
      # [filter_height, filter_width, in_channels, out_channels]
      W_fc1, b_fc1     = self._fc_weight_variable([51, 100], "fc1")
      W_fc2, b_fc2     = self._fc_weight_variable([100, 32], "fc2")
      W_fc3, b_fc3     = self._fc_weight_variable([100, 32], "fc3")
      
      h_fc = tf.nn.relu(tf.matmul(y, W_fc1) + b_fc1)
      z_mean         = tf.tanh(tf.matmul(h_fc, W_fc2) + b_fc2)
      z_log_sigma_sq = tf.tanh(tf.matmul(h_fc, W_fc3) + b_fc3)
      return (z_mean, z_log_sigma_sq)

    
  def _create_generator_network(self, z, reuse=False):
    with tf.variable_scope("gen", reuse=reuse) as scope:
      W_fc1, b_fc1 = self._fc_weight_variable([32, 100], "fc1")
      W_fc2, b_fc2 = self._fc_weight_variable([100, 51], "fc2")

      h_fc1 = tf.nn.relu(tf.matmul(z,     W_fc1) + b_fc1)
      y_out = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
      return y_out

    
  def _create_network(self, vae):
    # tf Graph input
    self.x = tf.placeholder("float", shape=[None, 80, 80, 3])
    self.y = tf.placeholder("float", shape=[None, 51])
    
    with tf.variable_scope("scan"):
      self.z_mean, self.z_log_sigma_sq = self._create_recognition_network(self.y)
      self.z = self._sample_z(self.z_mean, self.z_log_sigma_sq)
      self.y_out = self._create_generator_network(self.z)

    with tf.variable_scope("vae", reuse=True):
      self.x_z_mean, self.x_z_log_sigma_sq = vae._create_recognition_network(self.x,
                                                                             reuse=True)
      self.x_z = self._sample_z(self.x_z_mean, self.x_z_log_sigma_sq)
      self.x_out = vae._create_generator_network(self.x_z, reuse=True)


  def _kl(self, mu1, log_sigma1_sq, mu2, log_sigma2_sq):
    return tf.reduce_sum(0.5 * (log_sigma2_sq - log_sigma1_sq +
                                tf.exp(log_sigma1_sq - log_sigma2_sq) +
                                tf.square(mu1 - mu2) / tf.exp(log_sigma2_sq) -
                                1), axis=-1)

  
  def _create_loss_optimizer(self):
    # Reconstruction loss
    reconstr_loss = 0.5 * tf.reduce_sum( tf.square(self.y - self.y_out) )

    # Latent loss
    latent_loss = self.beta * -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                                   - tf.square(self.z_mean) 
                                                   - tf.exp(self.z_log_sigma_sq))
    latent_loss2 = self.lambd * self._kl(self.x_z_mean, self.x_z_log_sigma_sq,
                                         self.z_mean, self.z_log_sigma_sq)

    self.loss = reconstr_loss + latent_loss + latent_loss2
    
    self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="scan")
    
    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate,
      epsilon=self.epsilon).minimize(self.loss, var_list=self.variables)

  
  def get_vars(self):
    return self.variables
