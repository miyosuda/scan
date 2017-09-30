# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf

class AutoEncoder(object):
  def __init__(self, learning_rate=1e-4):
    self.learning_rate = learning_rate
    
    # Create autoencoder network
    self._create_network()
    
    # Define loss function and corresponding optimizer
    self._create_loss_optimizer()

  def _conv_weight_variable(self, weight_shape, deconv=False):
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
    
    weight_initial = tf.random_uniform(weight_shape, minval=-d, maxval=d)
    bias_initial   = tf.random_uniform(bias_shape,   minval=-d, maxval=d)
    return tf.Variable(weight_initial), tf.Variable(bias_initial)
  
  
  def _fc_weight_variable(self, weight_shape):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight_initial = tf.random_uniform(weight_shape, minval=-d, maxval=d)
    bias_initial   = tf.random_uniform(bias_shape,   minval=-d, maxval=d)
    return tf.Variable(weight_initial), tf.Variable(bias_initial)
  
  
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
    #..output_shape = tf.pack([batch_size, out_height, out_width, out_channel])
    output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')
      
  
  def _create_recognition_network(self, x):
    # [filter_height, filter_width, in_channels, out_channels]
    W_conv1, b_conv1 = self._conv_weight_variable([4, 4, 3,  32])
    W_conv2, b_conv2 = self._conv_weight_variable([4, 4, 32, 32])
    W_conv3, b_conv3 = self._conv_weight_variable([4, 4, 32, 64])
    W_conv4, b_conv4 = self._conv_weight_variable([4, 4, 64, 64])
    W_fc1, b_fc1     = self._fc_weight_variable([5 * 5 * 64, 100])

    # TODO: selu
    
    h_conv1 = tf.nn.relu(self._conv2d(x, W_conv1, 2) + b_conv1)        # (40, 40)
    h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2, 2) + b_conv2)  # (20, 20)
    h_conv3 = tf.nn.relu(self._conv2d(h_conv2, W_conv3, 2) + b_conv3)  # (10, 10)
    h_conv4 = tf.nn.relu(self._conv2d(h_conv3, W_conv4, 2) + b_conv4)  # (5, 5)
    h_conv4_flat = tf.reshape(h_conv4, [-1, 5 * 5 * 64])
    z = tf.tanh(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
    return z

  
  def _create_generator_network(self, z):
    W_fc1, b_fc1 = self._fc_weight_variable([100, 5 * 5 * 64])
  
    # [filter_height, filter_width, output_channels, in_channels]
    W_deconv1, b_deconv1 = self._conv_weight_variable([4, 4, 64, 64],  deconv=True)
    W_deconv2, b_deconv2 = self._conv_weight_variable([4, 4, 32, 64], deconv=True)
    W_deconv3, b_deconv3 = self._conv_weight_variable([4, 4, 32, 32], deconv=True)
    W_deconv4, b_deconv4 = self._conv_weight_variable([4, 4, 3, 32],  deconv=True)

    # TODO: selu?
  
    h_fc1 = tf.nn.relu(tf.matmul(z, W_fc1) + b_fc1)
    h_fc1_reshaped = tf.reshape(h_fc1, [-1, 5, 5, 64])
    h_deconv1 = tf.tanh(self._deconv2d(h_fc1_reshaped, W_deconv1, 5, 5, 2) + b_deconv1)
    h_deconv2 = tf.tanh(self._deconv2d(h_deconv1, W_deconv2, 10, 10, 2) + b_deconv2)
    h_deconv3 = tf.tanh(self._deconv2d(h_deconv2, W_deconv3, 20, 20, 2) + b_deconv3)
    y = tf.sigmoid(self._deconv2d(h_deconv3, W_deconv4, 40, 40, 2) + b_deconv4)
    return y


  def _create_network(self):
    # tf Graph input
    self.x = tf.placeholder("float", shape=[None, 80, 80, 3])
    self.z = self._create_recognition_network(self.x)
    self.y = self._create_generator_network(self.z)

    
  def _create_loss_optimizer(self):
    # Reconstruction loss
    reconstr_loss = 0.5 * tf.reduce_sum( tf.square(self.x - self.y) )
    self.cost = reconstr_loss

    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate).minimize(self.cost)


  def partial_fit(self, sess, X):
    """Train model based on mini-batch of input data.
    
    Return cost of mini-batch.
    """
    opt, cost = sess.run((self.optimizer, self.cost), 
                         feed_dict={self.x: X})
    return cost


  def reconstruct(self, sess, X):
    """ Reconstruct given data. """
    return sess.run(self.y, 
                    feed_dict={self.x: X})
