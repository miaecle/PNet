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

class ResidueEmbedding(Layer):

  def __init__(self,
               pos_start=0,
               pos_end=23,
               embedding_length=50,
               init='glorot_uniform',
               activation='relu',
               dropout=None,
               **kwargs):
    """
    Parameters
    ----------
    pos_start: int, optional
      starting position of raw features that need embedding
    pos_end: int, optional
      ending position
    embedding_length: int, optional
      length for embedding
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    dropout: float, optional
      Dropout probability, not supported here

    """
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.embedding_length = embedding_length
    super(ResidueEmbedding, self).__init__(**kwargs)

  def build(self):
    """ Construct internal trainable weights.
    """

    self.embedding = self.init([self.pos_end - self.pos_start, self.embedding_length])
    self.trainable_weights = [self.embedding]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()
    input_features = in_layers[0].out_tensor
    embedded_features = tf.tensordot(input_features[:, :, self.pos_start:self.pos_end], self.embedding, [[2], [0]])
    out_tensor = tf.concat([embedded_features, input_features[:, :, self.pos_end:]], axis=2)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

class Conv1DLayer(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size,
               init='glorot_uniform',
               activation='relu',
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
    super(Conv1DLayer, self).__init__(**kwargs)

  def build(self):
    """ Construct internal trainable weights.
    """

    self.W = self.init([self.n_size, self.n_input_feat, self.n_output_feat])
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
    out_tensor = tf.nn.conv1d(input_features, self.W, stride=1, padding='SAME')
    out_tensor = tf.nn.bias_add(out_tensor, self.b)
    if len(in_layers) > 1:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=2)
      out_tensor = out_tensor * tf.to_float(flag)
    out_tensor = self.activation(out_tensor)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor


class Conv2DLayer(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size,
               init='glorot_uniform',
               activation='relu',
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
    super(Conv2DLayer, self).__init__(**kwargs)

  def build(self):
    """ Construct internal trainable weights.
    """

    self.W = self.init([self.n_size, self.n_size, self.n_input_feat, self.n_output_feat])
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
    out_tensor = tf.nn.conv2d(input_features, self.W, strides=[1, 1, 1, 1], padding='SAME')
    out_tensor = tf.nn.bias_add(out_tensor, self.b)
    if len(in_layers) > 1:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=3)
      out_tensor = out_tensor * tf.to_float(flag)
    out_tensor = self.activation(out_tensor)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

class Outer1DTo2DLayer(Layer):

  def __init__(self,
               batch_size,
               max_n_res=1000,
               pad_length=25,
               n_features=50,
               **kwargs):
    self.max_n_res = max_n_res
    self.pad_length = pad_length
    self.batch_size = batch_size
    self.n_features = n_features
    super(Outer1DTo2DLayer, self).__init__(**kwargs)

  def build(self):
    """ Construct internal trainable weights.
    """
    self.trainable_weights = []

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features, input_flag_2D
    This layer concats 1D sequences into 2D sequences
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    self.build()
    max_n_res = self.max_n_res
    pad_length = self.pad_length
    n_features = self.n_features
    batch_size = self.batch_size

    input_features = in_layers[0].out_tensor
    n_residues = in_layers[2].out_tensor + pad_length

    pad_remains = max_n_res - n_residues
    features_separated = tf.split(input_features, n_residues, axis=1)
    features_padded = [tf.pad(features_separated[i],
                              [[0,0],[0,pad_remains[i]],[0,0]],
                              mode='CONSTANT') for i in range(batch_size)]
    features_padded = [tf.reshape(
        tf.expand_dims(tf.tile(features_padded[i],
                               [1, n_residues[i], 1]), axis=0),
        shape=[1, -1, max_n_res, n_features]) for i in range(batch_size)]

    tensor1 = tf.stack([input_features]*max_n_res, axis=2)
    tensor2 = tf.concat(features_padded, axis=1)

    out_tensor = [tensor1, tensor2]
    out_tensor = tf.concat(out_tensor, axis=3)

    if len(in_layers) > 1:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=3)
      out_tensor = out_tensor * tf.to_float(flag)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

class ContactMapGather(Layer):

  def __init__(self,
               n_input_feat,
               init='glorot_uniform',
               activation='relu',
               **kwargs):
    self.n_input_feat = n_input_feat
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    super(ContactMapGather, self).__init__(**kwargs)

  def build(self):
    """ Construct internal trainable weights.
    """
    self.W = self.init([self.n_input_feat, 2])
    self.b = model_ops.zeros(shape=[2])
    self.trainable_weights = [self.W, self.b]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features, input_flag_2D
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    self.build()
    input_features = in_layers[0].out_tensor
    input_features = tf.reshape(input_features, shape=[-1, self.n_input_feat])
    if len(in_layers) > 1:
      flag = tf.cast(tf.reshape(in_layers[1].out_tensor, [-1]), dtype=tf.bool)
      out_tensor = tf.boolean_mask(input_features, flag)
    out_tensor = tf.nn.xw_plus_b(out_tensor, self.W, self.b)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor