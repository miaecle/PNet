#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:07:15 2017

@author: zqwu
"""

import numpy as np
import tensorflow as tf
from deepchem.models.tensorgraph.layers import Input, Dense, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, \
    Weights, Feature, Squeeze
from pnet.models.layers import BatchNorm, ResidueEmbedding, Conv1DLayer, Conv2DLayer, \
    Outer1DTo2DLayer, ContactMapGather, ResAdd, Conv2DPool, Conv2DUp, \
    Expand_dim, ShapePool, ToShape
from pnet.models.conv_net_contact_map import to_one_hot, from_one_hot, ConvNetContactMapBase

class EncodeDecodeContactMap(ConvNetContactMapBase):
  def __init__(self,
               n_pool_layers=4,
               init_n_filter=32,
               **kwargs):
    """
    Parameters:
    -----------
    n_pool_layers: int, optional
      number of blocks(pooling layer) in the Conv2DModule
    init_n_filter: int, optional
      number of filters for the first Conv2D block
      after each pooling layer, n_filter is doubled
    """
    self.n_pool_layers = n_pool_layers
    self.init_n_filter = init_n_filter
    super(EncodeDecodeContactMap, self).__init__(**kwargs)

  def Conv1DModule(self, n_input, in_layer):

    self.conv_1D_layers = []
    self.batch_norm_layers = []
    self.res_layers = []

    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=self.init_n_filter,
        n_size=7,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name='global_conv_1'))
    
    in_layer = self.conv_1D_layers[-1]
    n_input = self.init_n_filter
    
    for i in range(3):
      # n_input = init_n_filter
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, size=17, name='Res1D_Module0_'+str(i)+'_')
    
    for i in range(3):
      # n_input = init_n_filter
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, size=7, name='Res1D_Module1_'+str(i)+'_')

    return n_input, in_layer

  def OuterModule(self, n_input, in_layer):
    # Add transform layer from 1D sequences to 2D sequences
    self.outer = Outer1DTo2DLayer(
        in_layers=[in_layer, self.n_residues, self.res_flag_2D, self.res_2D_features], name='global_outer')
    # n_input = init_n_filter*3 + 4 (100)
    n_input = n_input*3+self.n_res_2D_feat
    in_layer = self.outer
    return n_input, in_layer


  def Conv2DModule(self, n_input, in_layer):
    return self.encode_decode_structure(n_input, in_layer)

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

  def encode_decode_structure(self, n_input, in_layer):
  
    n_pool_layers = self.n_pool_layers
    self.shortcut_layers = []
    self.encode_pool_layers = []
    self.decode_up_layers = []
    self.res_flags = [self.res_flag_2D]
    self.shortcut_shapes = [ToShape(n_input, self.batch_size, in_layers=[self.n_residues], name='global_to_shape')]
    
    res_flag_2D = self.res_flag_2D
    
    for i in range(n_pool_layers):
      for j in range(3):
        # n_input = 100
        n_input, in_layer = self.Res2DModule_b(n_input, in_layer, res_flag_2D=res_flag_2D, name='Res2D_Encoding_Module_'+str(i)+'_Submodule_'+str(j)+'_')

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
      
      # n_input = 100
      n_input = n_input * 2
      n_input, in_layer = self.Res2DModule_c(n_input, in_layer, res_flag_2D=res_flag_2D, name='Res2D_Decoding_Module_'+str(j)+'_Down_')
      for i in range(2):
        # n_input = 100
        n_input, in_layer = self.Res2DModule_b(n_input, in_layer, res_flag_2D=res_flag_2D, name='Res2D_Decoding_Module_'+str(j)+'_Submodule_'+str(i)+'_')
      
    return n_input, in_layer
