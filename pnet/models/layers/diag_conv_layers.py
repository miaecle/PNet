#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:38:54 2017

@author: zqwu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from pnet.utils.tg_copy import activations
from pnet.utils.tg_copy import initializations
from pnet.utils.tg_copy import model_ops

from pnet.utils.tg_copy.layers import Layer
from pnet.utils.tg_copy.layers import convert_to_layers

class DiagConv2DLayer(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size,
               init='glorot_uniform',
               activation='relu',
               **kwargs):
    """
    Parameters
    ----------
    n_input_feat: int
      Number of input channels
    n_output_feat: int
      Number of output channels
    n_size: int
      Number of filter size(full length)
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied

    """
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.n_input_feat = n_input_feat
    self.n_output_feat = n_output_feat
    self.n_size = n_size
    assert self.n_size % 2 == 1, "filter size needs to be odd"
    super(DiagConv2DLayer, self).__init__(**kwargs)

  def build(self):
    """ Construct internal trainable weights.
    """
    self.W = []
    n_size = self.n_size
    for i in range(n_size):
      if i < np.floor(n_size/2):
        W_effective = tf.Variable(tf.zeros((n_size, self.n_input_feat, self.n_output_feat)), trainable=False)
      else:
        W_trainable = self.init([n_size - i, self.n_input_feat, self.n_output_feat])
        W_before = tf.Variable(tf.zeros((i - int(np.floor(n_size/2.)), self.n_input_feat, self.n_output_feat)), trainable=False)
        W_after = tf.Variable(tf.zeros((int(np.floor(n_size/2.)), self.n_input_feat, self.n_output_feat)), trainable=False)
        W_effective = tf.concat([W_before, W_trainable, W_after], 0)
      self.W.append(W_effective)

    self.W = tf.stack(self.W, axis=0)
    self.b = model_ops.zeros((self.n_output_feat,))
    self.trainable_weights = [self.W, self.b]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features, input_flag
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()

    input_features = in_layers[0].out_tensor
    if self.activation_first:
      input_features = self.activation(input_features)
    out_tensor = tf.nn.conv2d(input_features, self.W, strides=[1, 1, 1, 1], padding='SAME')
    out_tensor = tf.nn.bias_add(out_tensor, self.b)
    if len(in_layers) > 1:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=3)
      out_tensor = out_tensor * tf.to_float(flag)
    if not self.activation_first:
      out_tensor = self.activation(out_tensor)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

class DiagConv2DAtrous(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size,
               rate,
               init='glorot_uniform',
               activation='relu',
               activation_first=True,
               dropout=None,
               **kwargs):
    """
    Parameters
    ----------
    n_input_feat: int
      Number of input channels
    n_output_feat: int
      Number of output channels
    n_size: int
      Number of filter size(full length)
    rate: int
      Rate of atrous convolution
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    dropout: float, optional
      Dropout probability, not supported here

    """
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.n_input_feat = n_input_feat
    self.n_output_feat = n_output_feat
    self.n_size = n_size
    self.rate = rate
    self.activation_first = activation_first
    super(DiagConv2DAtrous, self).__init__(**kwargs)

  def build(self):
    """ Construct internal trainable weights.
    """

    self.W = []
    n_size = self.n_size
    for i in range(n_size):
      if i < np.floor(n_size/2):
        W_effective = tf.Variable(tf.zeros((n_size, self.n_input_feat, self.n_output_feat)), trainable=False)
      else:
        W_trainable = self.init([n_size - i, self.n_input_feat, self.n_output_feat])
        W_before = tf.Variable(tf.zeros((i - int(np.floor(n_size/2.)), self.n_input_feat, self.n_output_feat)), trainable=False)
        W_after = tf.Variable(tf.zeros((int(np.floor(n_size/2.)), self.n_input_feat, self.n_output_feat)), trainable=False)
        W_effective = tf.concat([W_before, W_trainable, W_after], 0)
      self.W.append(W_effective)

    self.W = tf.stack(self.W, axis=0)
    self.b = model_ops.zeros((self.n_output_feat,))
    self.trainable_weights = [self.W, self.b]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features, output_flag
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()

    input_features = in_layers[0].out_tensor

    if self.activation_first:
      input_features = self.activation(input_features)
    out_tensor = tf.nn.atrous_conv2d(input_features,
                                     self.W,
                                     rate=self.rate,
                                     padding='SAME')
    out_tensor = tf.nn.bias_add(out_tensor, self.b)
    if len(in_layers) > 1:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=3)
      out_tensor = out_tensor * tf.to_float(flag)
    if not self.activation_first:
      out_tensor = self.activation(out_tensor)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor
  
class DiagConv2DASPP(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size=5,
               rate=[6, 12, 18, 24],
               init='glorot_uniform',
               activation='relu',
               activation_first=True,
               **kwargs):
    """
    Parameters
    ----------
    n_input_feat: int
      Number of input channels
    n_output_feat: int
      Number of output channels for each Atrous component
    n_size: int
      Number of filter size(full length)
    rate: int
      Rate of each atrous convolution
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    activation_first: bool, optional
      If to apply activation before convolution

    """
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.n_input_feat = n_input_feat
    self.n_output_feat = n_output_feat
    self.n_size = n_size
    self.rate = rate
    self.activation_first = activation_first
    super(DiagConv2DASPP, self).__init__(**kwargs)

  def build(self):

    self.W = []
    n_size = self.n_size
    for i in range(n_size):
      if i < np.floor(n_size/2):
        W_effective = tf.Variable(tf.zeros((n_size, self.n_input_feat, self.n_output_feat)), trainable=False)
      else:
        W_trainable = self.init([n_size - i, self.n_input_feat, self.n_output_feat])
        W_before = tf.Variable(tf.zeros((i - int(np.floor(n_size/2.)), self.n_input_feat, self.n_output_feat)), trainable=False)
        W_after = tf.Variable(tf.zeros((int(np.floor(n_size/2.)), self.n_input_feat, self.n_output_feat)), trainable=False)
        W_effective = tf.concat([W_before, W_trainable, W_after], 0)
      self.W.append(W_effective)

    self.W = tf.stack(self.W, axis=0)
    self.b = model_ops.zeros((self.n_output_feat,))
    self.trainable_weights = [self.W, self.b]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features, output_flag
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()

    input_features = in_layers[0].out_tensor

    if self.activation_first:
      input_features = self.activation(input_features)
      
    n_channels = self.n_output_feat//len(self.rate)
    output_feats = [0]
    for i in range(1, len(self.rate)):
      output_feats.append(n_channels*i)
    output_feats.append(self.n_output_feat)
      
    out_tensors = [tf.nn.bias_add(tf.nn.atrous_conv2d(
        input_features, 
        self.W[:, :, :, output_feats[i]:output_feats[i+1]],
        rate=rate,
        padding='SAME'), self.b[output_feats[i]:output_feats[i+1]])
        for i, rate in enumerate(self.rate)]
    out_tensor = tf.concat(out_tensors, 3)
    
    if len(in_layers) > 1:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=3)
      out_tensor = out_tensor * tf.to_float(flag)
    if not self.activation_first:
      out_tensor = self.activation(out_tensor)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor