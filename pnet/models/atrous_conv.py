#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:07:15 2017

@author: zqwu
"""

import numpy as np
import tensorflow as tf
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Input, BatchNorm, Dense, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, \
    Weights, Feature, Squeeze
from pnet.models.layers import ResidueEmbedding, Conv1DLayer, Conv2DLayer, \
    Outer1DTo2DLayer, ContactMapGather, ResAdd, Conv2DPool, Conv2DUp, \
    Expand_dim, ShapePool, ToShape, Conv1DAtrous, Conv2DAtrous
from pnet.models.conv_net_contact_map import to_one_hot, from_one_hot, ConvNetContactMapBase

class AtrousConvContactMap(ConvNetContactMapBase):
  def __init__(self,
               filter_size_1D=[5]*4,
               n_filter_1D=[20]*4,
               filter_size_atrous_1D=[3]*4,
               n_filter_atrous_1D=[20]*4,
               n_pool_layers=2,
               filter_size_2D=3,
               init_n_filter=64,
               filter_size_atrous_2D=3,
               n_atrous=4,
               **kwargs):
    """
    Parameters:
    -----------
    filter_size_1D: list, optional
      structure of 1D convolution: size of convolution
    n_filter_1D: list, optional
      structure of 1D convolution: depths of convolution
    filter_size_atrous_1D: list, optional
      structure of atrous 1D convolution: size of convolution
    n_filter_atrous_1D: list, optional
      structure of atrous 1D convolution: depths of convolution
    n_pool_layers: int, optional
      number of blocks(pooling layer) in the Conv2DModule
    filter_size_2D: int, optional
      size of filter for Conv2DLayer
    init_n_filter: int, optional
      number of filters for the first Conv2D block
      after each pooling layer, n_filter is doubled
    filter_size_atrous_2D: int, optional
      size of filter for Conv2DAtrous
    n_atrous: int, optional
      number of atrous layers: rate increase from 2, 4, 8, ...
    """
    self.filter_size_1D = filter_size_1D
    self.n_filter_1D = n_filter_1D
    assert len(n_filter_1D) == len(filter_size_1D)
    self.filter_size_atrous_1D = filter_size_atrous_1D
    self.n_filter_atrous_1D = n_filter_atrous_1D
    assert len(n_filter_atrous_1D) == len(filter_size_atrous_1D)
    self.n_pool_layers = n_pool_layers
    self.filter_size_2D = filter_size_2D
    self.init_n_filter = init_n_filter
    self.filter_size_atrous_2D = filter_size_atrous_2D
    self.n_atrous = n_atrous
    super(AtrousConvContactMap, self).__init__(**kwargs)

  def Conv1DModule(self, n_input, in_layer):

    self.conv_1D_layers = []
    self.batch_norm_layers = []
    self.res_layers = []

    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=self.n_filter_1D[0],
        n_size=self.filter_size_1D[0],
        activation_first=False,
        in_layers=[in_layer, self.res_flag_1D]))
    n_input = self.n_filter_1D[0]
    in_layer = self.conv_1D_layers[-1]
    res_in = in_layer

    for i, layer_1D in enumerate(self.n_filter_1D):
      n_output = layer_1D
      self.conv_1D_layers.append(Conv1DLayer(
          n_input_feat=n_input,
          n_output_feat=n_output,
          n_size=self.filter_size_1D[i],
          in_layers=[in_layer, self.res_flag_1D]))
      self.batch_norm_layers.append(BatchNorm(in_layers=[self.conv_1D_layers[-1]]))
      n_input = n_output
      in_layer = self.batch_norm_layers[-1]
      if i%2 == 1:
        self.res_layers.append(ResAdd(in_layers=[in_layer, res_in]))
        in_layer = self.res_layers[-1]
        res_in = self.res_layers[-1]

    for i, layer_atrous_1D in enumerate(self.n_filter_atrous_1D):
      rate = 2**(i+1)
      n_output = layer_atrous_1D
      self.conv_1D_layers.append(Conv1DAtrous(
          n_input_feat=n_input,
          n_output_feat=n_output,
          n_size=self.filter_size_atrous_1D[i],
          rate = rate,
          in_layers=[in_layer, self.res_flag_1D]))
      self.batch_norm_layers.append(BatchNorm(in_layers=[self.conv_1D_layers[-1]]))
      n_input = n_output
      in_layer = self.batch_norm_layers[-1]

    return n_input, in_layer

  def OuterModule(self, n_input, in_layer):
    # Add transform layer from 1D sequences to 2D sequences
    self.outer = Outer1DTo2DLayer(
        in_layers=[in_layer, self.n_residues, self.res_flag_2D])
    n_input = n_input*2
    in_layer = self.outer
    return n_input, in_layer

  def Conv2DModule(self, n_input, in_layer):
    return self.atrous_structure(n_input, in_layer)

  def GatherModule(self, n_input, in_layer):
    # Transform all channels of a single contact to predicitons of contact probability
    if self.mode == "classification":
      n_output = 2
    elif self.mode == "regression":
      n_output = 1
    self.gather_layer = ContactMapGather(
        n_input_feat=n_input,
        n_output=n_output,
        in_layers=[in_layer, self.res_flag_2D])
    return n_output, self.gather_layer

    encode_decode_layer = self.encode_decode_structure(in_layer, n_input)
    decoded_layer, n_input = encode_decode_layer

  def atrous_structure(self, n_input, in_layer):

    n_pool_layers = self.n_pool_layers
    filter_size = self.filter_size_2D
    init_n_filter = self.init_n_filter

    self.encode_conv_layers = []
    self.shortcut_layers = []
    self.shortcut_shapes = [ToShape(init_n_filter,
                                    self.batch_size,
                                    in_layers=[self.n_residues])]
    self.encode_pool_layers = []
    self.decode_conv_layers = []
    self.decode_up_layers = []
    res_flag_2D = self.res_flag_2D
    self.res_flags = [self.res_flag_2D]
    for i in range(n_pool_layers):
      n_filter = init_n_filter * (2**i)
      self.encode_conv_layers.append(Conv2DLayer(
          n_input_feat=n_input,
          n_output_feat=n_filter,
          n_size=filter_size,
          in_layers=[in_layer, res_flag_2D]))
      self.batch_norm_layers.append(BatchNorm(in_layers=[self.encode_conv_layers[-1]]))
      in_layer = self.batch_norm_layers[-1]
      n_input = n_filter

      self.encode_conv_layers.append(Conv2DLayer(
          n_input_feat=n_input,
          n_output_feat=n_filter,
          n_size=filter_size,
          in_layers=[in_layer, res_flag_2D]))
      self.batch_norm_layers.append(BatchNorm(in_layers=[self.encode_conv_layers[-1]]))
      in_layer = self.batch_norm_layers[-1]
      n_input = n_filter

      self.shortcut_layers.append(in_layer)
      self.shortcut_shapes.append(ShapePool(in_layers=[self.shortcut_shapes[-1]]))

      self.encode_pool_layers.append(Conv2DPool(
          n_size=2,
          in_layers=[in_layer]))
      in_layer = self.encode_pool_layers[-1]

      flag_2D = Expand_dim(3, in_layers=[res_flag_2D])
      new_flag_2D = Conv2DPool(n_size=2, in_layers=[flag_2D])
      res_flag_2D = Squeeze(squeeze_dims=3, in_layers=[new_flag_2D])
      self.res_flags.append(res_flag_2D)


    n_filter = init_n_filter * (2**n_pool_layers)
    filter_size_atrous = self.filter_size_atrous_2D
    for i in range(self.n_atrous):
      self.encode_conv_layers.append(Conv2DAtrous(
          n_input_feat=n_input,
          n_output_feat=n_filter,
          n_size=filter_size_atrous,
          rate=2**(i+1),
          in_layers=[in_layer, res_flag_2D]))
      self.batch_norm_layers.append(BatchNorm(in_layers=[self.encode_conv_layers[-1]]))
      in_layer = self.batch_norm_layers[-1]
      n_input = n_filter


    for j in range(n_pool_layers):
      res_flag_2D = self.res_flags[-(j+2)]
      out_shape = self.shortcut_shapes[-(j+2)]
      self.decode_up_layers.append(Conv2DUp(
          n_input_feat=n_input,
          n_output_feat=n_input//2,
          n_size=2,
          in_layers=[in_layer, out_shape, res_flag_2D]
          ))
      in_layer = Concat(axis=3, in_layers=[self.decode_up_layers[-1],
          self.shortcut_layers[-(j+1)]])
      n_filter = init_n_filter * (2**(n_pool_layers-1-j))
      self.decode_conv_layers.append(Conv2DLayer(
          n_input_feat=n_input,
          n_output_feat=n_filter,
          n_size=filter_size,
          in_layers=[in_layer, res_flag_2D]))
      self.batch_norm_layers.append(BatchNorm(in_layers=[self.decode_conv_layers[-1]]))
      in_layer = self.batch_norm_layers[-1]
      n_input = n_filter

      self.decode_conv_layers.append(Conv2DLayer(
          n_input_feat=n_input,
          n_output_feat=n_filter,
          n_size=filter_size,
          in_layers=[in_layer, res_flag_2D]))
      self.batch_norm_layers.append(BatchNorm(in_layers=[self.decode_conv_layers[-1]]))
      in_layer = self.batch_norm_layers[-1]
      n_input = n_filter

    return n_input, in_layer