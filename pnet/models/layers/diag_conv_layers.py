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
from deepchem.nn import activations
from deepchem.nn import initializations
from deepchem.nn import model_ops

from deepchem.models.tensorgraph.layers import Layer
from deepchem.models.tensorgraph.layers import convert_to_layers

class DiagConv2DLayer(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size,
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
    assert self.n_size % 2 == 1, "filter size needs to be odd"
    self.activation_first = activation_first
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
                                     padding='VALID')
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