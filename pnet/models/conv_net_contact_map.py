#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:28:44 2017

@author: zqwu
"""

import numpy as np
import tensorflow as tf
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Input, BatchNorm, Dense, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, Weights, Feature
from pnet.models.layers import Conv1DLayer, Conv2DLayer, Outer1DTo2DLayer, ContactMapGather


class ConvNetContactMap(TensorGraph):
    def __init__(self,
                 n_res_feat,
                 batch_size,
                 filter_size_1D=[51, 25, 11],
                 n_filter_1D=[50, 50, 50],
                 filter_size_2D=[25, 25, 25],
                 n_filter_2D=[50, 50, 50],
                 max_n_res=1000,
                 **kwargs):
      self.n_res_feat = n_res_feat
      self.batch_size = batch_size
      self.filter_size_1D = filter_size_1D
      self.n_filter_1D = n_filter_1D
      assert len(n_filter_1D) == len(filter_size_1D)
      self.filter_size_2D = filter_size_2D
      self.n_filter_2D = n_filter_2D
      assert len(n_filter_2D) == len(filter_size_2D)
      self.max_n_res = max_n_res
      self.padding_length = int(np.ceil(max(filter_size_1D + filter_size_2D)/2.))
      super(ConvNetContactMap, self).__init__(**kwargs)
      self.build_graph()

    def build_graph(self):
      self.res_features = Feature(shape=(None, self.max_n_res, self.n_res_feat))
      self.res_flag_1D = Feature(shape=(None, self.max_n_res), dtype=tf.int32)
      self.res_flag_2D = Feature(shape=(None, self.max_n_res, self.max_n_res), dtype=tf.int32)
      self.n_res = Feature(shape=(self.batch_size), dtype=tf.int32)

      self.conv_1D_layers = []
      self.batch_norm_layers = []
      n_input = self.n_res_feat
      in_layer = self.res_features
      for i, layer_1D in self.n_filter_1D:
        n_output = layer_1D
        self.conv_1D_layers.append(Conv1DLayer(
          n_input_feat=n_input,
          n_output_feat=n_output,
          n_size=self.filter_size_1D[i],
          padding_length=self.padding_length,
          in_layers=[in_layer, self.res_flag_1D]))
        n_input = n_output
        in_layer = self.conv_1D_layers[-1]
        self.batch_norm_layers.append(BatchNorm(in_layers=[in_layer]))
        in_layer = self.batch_norm_layers[-1]

      self.outer = Outer1DTo2DLayer(
          max_n_res = self.max_n_res,
          in_layers=[in_layer, self.res_flag_2D])
      n_input = n_input*2
      in_layer = self.outer

      self.conv_2D_layers = []
      for i, layer_2D in self.n_filter_2D:
        n_output = layer_2D
        self.conv_2D_layers.append(Conv2DLayer(
          n_input_feat=n_input,
          n_output_feat=n_output,
          n_size=self.filter_size_2D[i],
          in_layers=[in_layer, self.res_flag_2D]))
        n_input = n_output
        in_layer = self.conv_2D_layers[-1]
        self.batch_norm_layers.append(BatchNorm(in_layers=[in_layer]))
        in_layer = self.batch_norm_layers[-1]

      self.gather_layer = ContactMapGather(
          n_input_feat=n_input,
          in_layers=[in_layer, self.res_flag_2D])

      softmax = SoftMax(in_layers=[self.gather_layer])
      self.add_output(softmax)

      self.label = Label(shape=(None, 2))
      self.weights = Weights(shape=(None,))
      cost = SoftMaxCrossEntropy(in_layers=[self.label, self.gather_layer])
      loss = WeightedError(in_layers=[cost, self.weights])
      self.set_loss(loss)
      return

    def fit(self):
      return

    def predict(self):
      return

    def predict_proba(self):
      return