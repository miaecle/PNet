#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:07:15 2017

@author: zqwu
"""

import numpy as np
import tensorflow as tf
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Input, Dense, Squeeze, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, \
    Weights, Feature
from pnet.models.layers import BatchNorm, ResidueEmbedding, Conv1DLayer, \
    Conv2DLayer, Outer1DTo2DLayer, ContactMapGather, ResAdd, Conv2DPool, \
    Conv2DUp, Expand_dim, ShapePool, ToShape, Conv1DAtrous, Conv2DAtrous, \
    Conv2DASPP, Conv1DLayer_RaptorX, Conv2DLayer_RaptorX
from pnet.models.conv_net_contact_map import to_one_hot, from_one_hot, ConvNetContactMapBase

class RaptorX_structure(ConvNetContactMapBase):
  def __init__(self,
               **kwargs):
    super(RaptorX_structure, self).__init__(**kwargs)
  
  
  def Conv1DModule(self, n_input, in_layer):

    self.conv_1D_layers = []
    self.batch_norm_layers = []
    self.res_layers = []

    res_in = in_layer    
    for i in range(3):
      self.conv_1D_layers.append(Conv1DLayer_RaptorX(
          n_input_feat=n_input,
          n_output_feat=60,
          n_size=17,
          in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name='Res1D_'+str(i)+'_conv_1'))
      in_layer = self.conv_1D_layers[-1]
      n_input = 60
      self.conv_1D_layers.append(Conv1DLayer_RaptorX(
          n_input_feat=n_input,
          n_output_feat=60,
          n_size=17,
          in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name='Res1D_'+str(i)+'_conv_2'))
      in_layer = self.conv_1D_layers[-1]
      n_input = 60
      self.res_layers.append(ResAdd(in_layers=[in_layer, res_in], name='Res1D_'+str(i)+'_res_add'))
      res_in = self.res_layers[-1]
      in_layer = self.res_layers[-1]

    return n_input, in_layer

  def OuterModule(self, n_input, in_layer):
    # Add transform layer from 1D sequences to 2D sequences
    self.outer = Outer1DTo2DLayer(
        in_layers=[in_layer, self.n_residues, self.res_flag_2D, self.res_2D_features], name='global_outer')
    # n_input = 52
    n_input = n_input*3+self.n_res_2D_feat
    in_layer = self.outer
    return n_input, in_layer

  def Conv2DModule(self, n_input, in_layer):
    self.conv_2D_layers = []
    self.conv_2D_layers.append(Conv2DLayer_RaptorX(
        n_input_feat=n_input,
        n_output_feat=60,
        n_size=3,
        in_layers=[in_layer, self.res_flag_2D, self.training_placeholder], name='Res2D_global'))
    
    in_layer = self.conv_2D_layers[-1]
    n_input = 60
    res_in = in_layer    
    
    for i in range(30):
      self.conv_2D_layers.append(Conv2DLayer_RaptorX(
          n_input_feat=n_input,
          n_output_feat=60,
          n_size=3,
          in_layers=[in_layer, self.res_flag_2D, self.training_placeholder], name='Res2D_'+str(i)+'_conv_1'))
      in_layer = self.conv_2D_layers[-1]
      n_input = 60
      self.conv_2D_layers.append(Conv2DLayer_RaptorX(
          n_input_feat=n_input,
          n_output_feat=60,
          n_size=3,
          in_layers=[in_layer, self.res_flag_2D, self.training_placeholder], name='Res2D_'+str(i)+'_conv_2'))
      in_layer = self.conv_2D_layers[-1]
      n_input = 60
      
      self.res_layers.append(ResAdd(in_layers=[in_layer, res_in], name='Res2D_'+str(i)+'_res_add'))
      res_in = self.res_layers[-1]
      in_layer = self.res_layers[-1]

    return n_input, in_layer

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
