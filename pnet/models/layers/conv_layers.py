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
               pos_end=25,
               embedding_length=50,
               init='glorot_uniform',
               activation='relu',
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
    """
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.embedding_length = embedding_length
    super(ResidueEmbedding, self).__init__(**kwargs)

  def build(self):
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

    i = tf.shape(input_features)[0]
    j = tf.shape(input_features)[1]
    embedded_features = tf.reshape(tf.matmul(tf.reshape(input_features[:, :, self.pos_start:self.pos_end],
                                                        [i*j, self.pos_end - self.pos_start]),
                                             self.embedding),
                                   [i, j, self.embedding_length])

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
               activation_first=True,
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
    activation_first: bool, optional
      If to apply activation before convolution

    """
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.activation_first = activation_first # If to switch the order of convolution and activation
    self.n_input_feat = n_input_feat
    self.n_output_feat = n_output_feat
    self.n_size = n_size
    super(Conv1DLayer, self).__init__(**kwargs)

  def build(self):
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
    if self.activation_first:
      input_features = self.activation(input_features)
    out_tensor = tf.nn.conv1d(input_features, self.W, stride=1, padding='SAME')
    out_tensor = tf.nn.bias_add(out_tensor, self.b)
    if len(in_layers) > 1:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=2)
      out_tensor = out_tensor * tf.to_float(flag)
    if not self.activation_first:
      out_tensor = self.activation(out_tensor)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor


class Conv1DAtrous(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size,
               rate,
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
      Number of output channels
    n_size: int
      Number of filter size(full length)
    rate: int
      Rate of atrous convolution
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied
    activation_first: bool, optional
      If to apply activation before convolution

    """
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.activation_first = activation_first # If to switch the order of convolution and activation
    self.n_input_feat = n_input_feat
    self.n_output_feat = n_output_feat
    self.n_size = n_size
    self.rate = rate
    super(Conv1DAtrous, self).__init__(**kwargs)

  def build(self):
    self.W_effective = self.init([self.n_size, self.n_input_feat, self.n_output_feat])
    self.W = []
    for i in range(self.n_size):
      self.W.append(self.W_effective[i:i+1, :, :])
      if i < self.n_size - 1:
        self.W.append(tf.Variable(tf.zeros((self.rate-1, self.n_input_feat, self.n_output_feat)), trainable=False))
    self.W = tf.concat(self.W, 0)
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
    out_tensor = tf.nn.conv1d(input_features, self.W, stride=1, padding='SAME')
    out_tensor = tf.nn.bias_add(out_tensor, self.b)
    if len(in_layers) > 1:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=2)
      out_tensor = out_tensor * tf.to_float(flag)
    if not self.activation_first:
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
               activation_first=True,
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
    activation_first: bool, optional
      If to apply activation before convolution

    """
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    self.n_input_feat = n_input_feat
    self.n_output_feat = n_output_feat
    self.n_size = n_size
    self.activation_first = activation_first
    super(Conv2DLayer, self).__init__(**kwargs)

  def build(self):
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

class Conv2DAtrous(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size,
               rate,
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
      Number of output channels
    n_size: int
      Number of filter size(full length)
    rate: int
      Rate of atrous convolution
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
    super(Conv2DAtrous, self).__init__(**kwargs)

  def build(self):
    self.W = self.init([self.n_size, self.n_size, self.n_input_feat, self.n_output_feat])
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

class Conv2DASPP(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size=3,
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
    super(Conv2DASPP, self).__init__(**kwargs)

  def build(self):
    self.W = self.init([self.n_size, self.n_size, self.n_input_feat, self.n_output_feat])
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
  
class Outer1DTo2DLayer(Layer):

  def __init__(self,
               **kwargs):
    super(Outer1DTo2DLayer, self).__init__(**kwargs)

  def build(self):
    self.trainable_weights = []

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features, input_flag_2D
    This layer concats 1D sequences into 2D sequences
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    self.build()
    input_features = in_layers[0].out_tensor
    max_n_res = tf.reduce_max(in_layers[1].out_tensor, keep_dims=True)
    tensor1 = tf.expand_dims(input_features, axis=1)
    max_n_res1 = tf.concat([tf.constant([1]), max_n_res, tf.constant([1]), tf.constant([1])], axis=0)
    tensor1 = tf.tile(tensor1, max_n_res1)
    tensor2 = tf.expand_dims(input_features, axis=2)
    max_n_res2 = tf.concat([tf.constant([1]), tf.constant([1]), max_n_res, tf.constant([1])], axis=0)
    tensor2 = tf.tile(tensor2, max_n_res2)
    out_tensor = tf.concat([tensor1, tensor2], axis=3)

    if len(in_layers) > 2:
      flag = tf.expand_dims(in_layers[2].out_tensor, axis=3)
      out_tensor = out_tensor * tf.to_float(flag)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

class ContactMapGather(Layer):

  def __init__(self,
               n_input_feat,
               n_output=2,
               init='glorot_uniform',
               activation='relu',
               **kwargs):
    """
    Parameters
    ----------
    n_input_feat: int
      Number of input channels
    n_output: int, optional
      Number of output channels: 2 for classification, 1 for regression
    init: str, optional
      Weight initialization for filters.
    activation: str, optional
      Activation function applied

    """

    self.n_input_feat = n_input_feat
    self.n_output = n_output
    self.init = initializations.get(init)  # Set weight initialization
    self.activation = activations.get(activation)  # Get activations
    super(ContactMapGather, self).__init__(**kwargs)

  def build(self):
    self.W = self.init([self.n_input_feat, self.n_output])
    self.b = model_ops.zeros(shape=[self.n_output])
    self.trainable_weights = [self.W, self.b]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features, input_flag_2D
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    self.build()
    input_features = in_layers[0].out_tensor
    input_features = self.activation(input_features)
    input_features = (input_features + tf.transpose(input_features, perm=[0, 2, 1, 3])) / 2
    if len(in_layers) > 1:
      flag = tf.cast(in_layers[1].out_tensor, dtype=tf.bool)
      out_tensor = tf.boolean_mask(input_features, flag)
    out_tensor = tf.nn.xw_plus_b(out_tensor, self.W, self.b)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

class ResAdd(Layer):

  def __init__(self,
               fx_in_channels=None,
               x_in_channels=None,
               **kwargs):
    """
    Parameters
    ----------
    fx_in_channels: int(or None), optional
      Number of channels for fx: automatically defined if None
    x_in_channels: int(or None), optional
      Number of channels for x: automatically defined if None

    """
    self.fx_in_channels = fx_in_channels
    self.x_in_channels = x_in_channels
    super(ResAdd, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: fx, x
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    fx = in_layers[0].out_tensor
    x = in_layers[1].out_tensor

    pad_dimension = len(x.get_shape()) - 1
    if self.fx_in_channels is None:
      self.fx_in_channels = fx.get_shape().as_list()[-1]
    if self.x_in_channels is None:
      self.x_in_channels = x.get_shape().as_list()[-1]

    pad_length = self.fx_in_channels - self.x_in_channels
    assert pad_length >= 0

    pad = [[0,0]] * pad_dimension + [[0, pad_length]]
    out_tensor = fx + tf.pad(x, pad, "CONSTANT")
    if set_tensors:
      self.variables = None
      self.out_tensor = out_tensor
    return out_tensor

class Conv2DPool(Layer):

  def __init__(self,
               n_size=2,
               **kwargs):
    """
    Parameters
    ----------
    n_size: int
      Number of filter size(full length)

    """
    self.n_size = n_size
    super(Conv2DPool, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    input_features = in_layers[0].out_tensor
    out_tensor = tf.nn.max_pool(input_features, [1, self.n_size, self.n_size, 1],
                                strides=[1, self.n_size, self.n_size, 1], padding='SAME')
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class Conv2DUp(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size,
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
      Number of output channels
    n_size: int
      Number of filter size(full length)
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
    self.activation_first = activation_first
    super(Conv2DUp, self).__init__(**kwargs)

  def build(self):
    self.W = self.init([self.n_size, self.n_size, self.n_output_feat, self.n_input_feat])
    self.b = model_ops.zeros((self.n_output_feat,))
    self.trainable_weights = [self.W, self.b]

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features, output_shape, output_flag
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    self.build()

    input_features = in_layers[0].out_tensor
    out_shape = in_layers[1].out_tensor

    if self.activation_first:
      input_features = self.activation(input_features)
    out_tensor = tf.nn.conv2d_transpose(input_features,
                                        self.W,
                                        out_shape,
                                        strides=[1, self.n_size, self.n_size, 1],
                                        padding='SAME')
    out_tensor = tf.nn.bias_add(out_tensor, self.b)
    if len(in_layers) > 2:
      flag = tf.expand_dims(in_layers[2].out_tensor, axis=3)
      out_tensor = out_tensor * tf.to_float(flag)
    if not self.activation_first:
      out_tensor = self.activation(out_tensor)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

class Conv2DBilinearUp(Layer):

  def __init__(self,
               uppertri=False,
               **kwargs):
    self.uppertri = uppertri
    super(Conv2DBilinearUp, self).__init__(**kwargs)
    
  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    """ parent layers: input_features, output_shape, output_flag
    """
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    input_features = in_layers[0].out_tensor
    if self.uppertri:
      diag_elems = tf.transpose(tf.matrix_band_part(tf.transpose(input_features, 
                                                                 perm=[0,3,1,2]),
                                                    0, 0), 
                                perm=[0,2,3,1])
      input_features = input_features + tf.transpose(input_features, perm=[0,2,1,3]) - diag_elems
    
    out_shape = in_layers[1].out_tensor
    out_tensor = tf.image.resize_bilinear(input_features, out_shape[1:3])
    if len(in_layers) > 2:
      flag = tf.expand_dims(in_layers[2].out_tensor, axis=3)
      out_tensor = out_tensor * tf.to_float(flag)
      
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor