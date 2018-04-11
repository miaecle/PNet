#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:07:15 2017

@author: zqwu
"""

import numpy as np
import tensorflow as tf
from pnet.utils.tg_copy.tensor_graph import TensorGraph
from pnet.utils.tg_copy.layers import Input, Dense, Squeeze, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, \
    Weights, Feature
from pnet.models.layers import BatchNorm, ResidueEmbedding, Conv1DLayer, \
    Conv2DLayer, Outer1DTo2DLayer, ContactMapGather, ResAdd, Conv2DPool, \
    Conv2DUp, Expand_dim, ShapePool, ToShape, Conv1DAtrous, Conv2DAtrous, \
    Conv2DASPP

from pnet.utils.tg_copy.layers import Layer
from pnet.utils.tg_copy.layers import convert_to_layers
from pnet.utils.tg_copy import activations
from pnet.utils.tg_copy import initializations
from pnet.utils.tg_copy import model_ops

from pnet.models.conv_net_contact_map import to_one_hot, from_one_hot, ConvNetContactMapBase

class CustomConv2DLayer(Layer):

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
    super(CustomConv2DLayer, self).__init__(**kwargs)

  def build(self):
    n_size = self.n_size
    W_trainable = self.init([(n_size+1)*n_size/2, self.n_input_feat, self.n_output_feat])
    inds = np.ones((n_size, n_size))*-1
    j = 0
    for i in range(n_size):
      inds[i, :(n_size-i)] = np.array(range(j, j+n_size-i))
      j = j+n_size-i
      for k in range(i):
        inds[i, -1-k] = inds[k, -1-i]

    self.W = tf.gather(W_trainable, inds.astype(int))
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
    
    if len(in_layers) > 2:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=3)
      train_flag = in_layers[2].out_tensor
      out_tensor = tf.layers.batch_normalization(out_tensor, training=train_flag)
      out_tensor = out_tensor * tf.to_float(flag)
    elif len(in_layers) > 1:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=3)
      out_tensor = out_tensor * tf.to_float(flag)
    
    out_tensor = self.activation(out_tensor)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

class CustomConv2DAtrous(Layer):

  def __init__(self,
               n_input_feat,
               n_output_feat,
               n_size,
               rate,
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
    super(CustomConv2DAtrous, self).__init__(**kwargs)

  def build(self):
    n_size = self.n_size
    W_trainable = self.init([(n_size+1)*n_size/2, self.n_input_feat, self.n_output_feat])
    inds = np.ones((n_size, n_size))*-1
    j = 0
    for i in range(n_size):
      inds[i, :(n_size-i)] = np.array(range(j, j+n_size-i))
      j = j+n_size-i
      for k in range(i):
        inds[i, -1-k] = inds[k, -1-i]

    self.W = tf.gather(W_trainable, inds.astype(int))
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

    out_tensor = tf.nn.atrous_conv2d(input_features,
                                     self.W,
                                     rate=self.rate,
                                     padding='SAME')
    out_tensor = tf.nn.bias_add(out_tensor, self.b)
    
    if len(in_layers) > 2:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=3)
      train_flag = in_layers[2].out_tensor
      out_tensor = tf.layers.batch_normalization(out_tensor, training=train_flag)
      out_tensor = out_tensor * tf.to_float(flag)
    elif len(in_layers) > 1:
      flag = tf.expand_dims(in_layers[1].out_tensor, axis=3)
      out_tensor = out_tensor * tf.to_float(flag)
      
    out_tensor = self.activation(out_tensor)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor

class CustomAtrousConvContactMap(ConvNetContactMapBase):
  def __init__(self,
               filter_size=3,
               **kwargs):
    self.filter_size = filter_size
    super(CustomAtrousConvContactMap, self).__init__(**kwargs)
  
  def Res1DAtrousModule_b(self, n_input, in_layer, rate=2, size=3, name=None):
    if name == None:
      name = 'Res1DAtrous_same_'+str(self.module_count)+'_'
      self.module_count += 1
    
    in_layer_branch1 = in_layer
    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    self.conv_1D_layers.append(Conv1DAtrous(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        rate=rate,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    self.conv_1D_layers.append(Conv1DAtrous(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        rate=rate,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_1D_layers[-1]

    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input,
        n_size=1,
        in_layers=[in_layer_branch2, self.res_flag_1D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_1D_layers[-1]
    
    n_output = n_input
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def CustomRes2DAtrousModule_b(self, n_input, in_layer, rate=2, res_flag_2D=None, size=3, name=None):
    if name == None:
      name = 'CustomRes2DAtrous_same_'+str(self.module_count)+'_'
      self.module_count += 1
    
    if res_flag_2D is None:
      res_flag_2D = self.res_flag_2D
    
    in_layer_branch1 = in_layer
    
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(CustomConv2DAtrous(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        rate=rate,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(CustomConv2DAtrous(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        rate=rate,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]

    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def CustomRes2DModule_a(self, n_input, in_layer, res_flag_2D=None, size=3, name=None):
    if name == None:
      name = 'CustomRes2D_up_'+str(self.module_count)+'_'
      self.module_count += 1
    
    if res_flag_2D is None:
      res_flag_2D = self.res_flag_2D
      
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input*2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_b1'))
    in_layer_branch1 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]

    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]

    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input*2,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input * 2
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def CustomRes2DModule_b(self, n_input, in_layer, res_flag_2D=None, size=3, name=None):
    if name == None:
      name = 'CustomRes2D_same_'+str(self.module_count)+'_'
      self.module_count += 1
    
    if res_flag_2D is None:
      res_flag_2D = self.res_flag_2D
    
    in_layer_branch1 = in_layer
    
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]

    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer

  def CustomRes2DModule_c(self, n_input, in_layer, res_flag_2D=None, size=3, name=None):
    if name == None:
      name = 'CustomRes2D_down_'+str(self.module_count)+'_'
      self.module_count += 1
    
    if res_flag_2D is None:
      res_flag_2D = self.res_flag_2D
    
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_b1'))
    in_layer_branch1 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer, res_flag_2D, self.training_placeholder], name=name+'conv_a1'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a2'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=size,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a3'))
    in_layer_branch2 = self.conv_2D_layers[-1]

    self.conv_2D_layers.append(CustomConv2DLayer(
        n_input_feat=n_input//2,
        n_output_feat=n_input//2,
        n_size=1,
        in_layers=[in_layer_branch2, res_flag_2D, self.training_placeholder], name=name+'conv_a4'))
    in_layer_branch2 = self.conv_2D_layers[-1]
    
    n_output = n_input//2
    self.res_layers.append(ResAdd(in_layers=[in_layer_branch1, in_layer_branch2], name=name+'res_add'))
    out_layer = self.res_layers[-1]
    return n_output, out_layer
    

  def Conv1DModule(self, n_input, in_layer):

    self.conv_1D_layers = []
    self.batch_norm_layers = []
    self.res_layers = []

    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=32,
        n_size=7,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name='global_conv_1'))
    
    in_layer = self.conv_1D_layers[-1]
    n_input = 32
    
    for i in range(3):
      # n_input = 32
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, size=7, name='Res1D_'+str(i)+'_')
    
    for i in range(2):
      # n_input = 32
      n_input, in_layer = self.Res1DAtrousModule_b(n_input, in_layer, rate=2, name='Res1DAtrous_'+str(i)+'_rate'+str(2)+'_')
    n_input, in_layer = self.Res1DAtrousModule_b(n_input, in_layer, rate=4, name='Res1DAtrous_rate'+str(4)+'_')

    return n_input, in_layer

  def OuterModule(self, n_input, in_layer):
    # Add transform layer from 1D sequences to 2D sequences
    self.outer = Outer1DTo2DLayer(
        in_layers=[in_layer, self.n_residues, self.res_flag_2D, self.res_2D_features], name='global_outer')
    # n_input = 100
    n_input = n_input*3+self.n_res_2D_feat
    in_layer = self.outer
    return n_input, in_layer

  def Conv2DModule(self, n_input, in_layer):
    return self.custom_atrous_structure(n_input, in_layer)

  def GatherModule(self, n_input, in_layer, n_output=None):
    # Transform all channels of a single contact to predicitons of contact probability
    if n_output is None:
      if self.mode == "classification":
        n_output = 2
      elif self.mode == "regression":
        n_output = 1
    self.gather_layer = ContactMapGather(
        n_input_feat=n_input,
        n_output=n_output,
        in_layers=[in_layer, self.res_flag_2D], name='global_gather')
    return n_output, self.gather_layer

  def custom_atrous_structure(self, n_input, in_layer):

    n_pool_layers = 2

    self.shortcut_layers = []
    self.encode_pool_layers = []
    self.decode_up_layers = []
    self.res_flags = [self.res_flag_2D]
    self.shortcut_shapes = [ToShape(n_input, self.batch_size, in_layers=[self.n_residues], name='global_to_shape')]
    
    res_flag_2D = self.res_flag_2D
    for i in range(n_pool_layers):
      for j in range(4):
        # n_input = 100
        n_input, in_layer = self.CustomRes2DModule_b(n_input, in_layer, res_flag_2D=res_flag_2D, size=self.filter_size, name='Res2D_Encoding_Module_'+str(i)+'_Submodule_'+str(j)+'_')

      self.shortcut_layers.append(in_layer)

      self.encode_pool_layers.append(Conv2DPool(
          n_size=2,
          in_layers=[in_layer], name='global_pool_'+str(i)))
      in_layer = self.encode_pool_layers[-1]

      flag_2D = Expand_dim(3, in_layers=[res_flag_2D], name='global_oldflag_'+str(i))
      new_flag_2D = Conv2DPool(n_size=2, in_layers=[flag_2D], name='global_newflag_'+str(i))
      res_flag_2D = Squeeze(squeeze_dims=3, in_layers=[new_flag_2D], name='global_flag_'+str(i))
      self.res_flags.append(res_flag_2D)
      self.shortcut_shapes.append(ShapePool(n_filter=n_input, in_layers=[self.shortcut_shapes[-1]], name='global_shape_pool_'+str(i)))
      

    for i in range(3):
      # n_input = 100
      n_input, in_layer = self.CustomRes2DAtrousModule_b(n_input, in_layer, rate=2, res_flag_2D=res_flag_2D, size=self.filter_size, name='Res2D_Inter_Module_'+str(i)+'_Rate_'+str(2)+'_')
    n_input, in_layer = self.CustomRes2DAtrousModule_b(n_input, in_layer, rate=4, res_flag_2D=res_flag_2D, size=self.filter_size, name='Res2D_Inter_Module_'+str(2)+'_Rate_'+str(4)+'_')
    n_input, in_layer = self.CustomRes2DAtrousModule_b(n_input, in_layer, rate=4, res_flag_2D=res_flag_2D, size=self.filter_size, name='Res2D_Inter_Module_'+str(2)+'_Rate_'+str(4)+'_')
    n_input, in_layer = self.CustomRes2DAtrousModule_b(n_input, in_layer, rate=8, res_flag_2D=res_flag_2D, size=self.filter_size, name='Res2D_Inter_Module_'+str(3)+'_Rate_'+str(8)+'_')
    n_input, in_layer = self.CustomRes2DAtrousModule_b(n_input, in_layer, rate=8, res_flag_2D=res_flag_2D, size=self.filter_size, name='Res2D_Inter_Module_'+str(3)+'_Rate_'+str(8)+'_')
    n_input, in_layer = self.CustomRes2DAtrousModule_b(n_input, in_layer, rate=16, res_flag_2D=res_flag_2D, size=self.filter_size, name='Res2D_Inter_Module_'+str(4)+'_Rate_'+str(16)+'_')
    
    in_layer = self.conv_2D_layers[-1]
    
    for j in range(n_pool_layers):
      res_flag_2D = self.res_flags[-(j+2)]
      out_shape = self.shortcut_shapes[-(j+2)]
      self.decode_up_layers.append(Conv2DUp(
          n_input_feat=n_input,
          n_output_feat=n_input,
          n_size=2,
          in_layers=[in_layer, out_shape, res_flag_2D],
          name='global_upconv_'+str(j)))
      
      # n_input = 200
      in_layer = Concat(axis=3, in_layers=[self.decode_up_layers[-1],
          self.shortcut_layers[-(j+1)]], name='global_upconcat_'+str(j))
      
      n_input = n_input * 2
      n_input, in_layer = self.CustomRes2DModule_c(n_input, in_layer, res_flag_2D=res_flag_2D, size=self.filter_size, name='Res2D_Decoding_Module_'+str(j)+'_Down_')
      for i in range(3):
        # n_input = 100
        n_input, in_layer = self.CustomRes2DModule_b(n_input, in_layer, res_flag_2D=res_flag_2D, size=self.filter_size, name='Res2D_Decoding_Module_'+str(j)+'_Submodule_'+str(i)+'_')
      
        
    return n_input, in_layer
