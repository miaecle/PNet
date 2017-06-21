#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:28:44 2017

@author: zqwu
"""

import numpy as np
import tensorflow as tf
import deepchem
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Input, BatchNorm, Dense, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, Weights, Feature
    
class ConvNetContactMap(TensorGraph):
    def __init__(self, 
                 n_res_feat, 
                 filter_size_1D=[51, 25, 11], 
                 n_filter_1D=[50, 50, 50], 
                 filter_size_2D=[25, 25, 25],
                 n_filter_2D=[50, 50, 50], 
                 **kwargs):
      self.n_res_feat = n_res_feat
      self.filter_size_1D = filter_size_1D
      self.n_filter_1D = n_filter_1D
      self.filter_size_2D = filter_size_2D
      self.n_filter_2D = n_filter_2D
      super(ConvNetContactMap, self).__init__(**kwargs)
      self.build_graph()
      
    def build_graph(self):
      self.res_features = Feature(shape=(None, self.n_res_feat))
      
      return
    
    def fit(self):
      return
    
    def predict(self):
      return
    
    def predict_proba(self):
      return