#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:28:44 2017
@author: zqwu
"""

import numpy as np
import tensorflow as tf
import pickle
import time
import threading

import deepchem as dc
from rdkit import Chem
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Input, Dense, \
    SoftMax, SoftMaxCrossEntropy, L2Loss, Concat, WeightedError, Label, \
    Weights, Feature, TensorWrapper, GraphConv, GraphPool, GraphGather, Add, \
    Reshape, Squeeze
    
from deepchem.models.tensorgraph.optimizers import Adam
from pnet.models.layers import BatchNorm, AminoAcidEmbedding, AminoAcidPad, \
    Conv1DLayer, Conv2DLayer, Outer1DTo2DLayer, ContactMapGather, ResAdd, \
    WeightedL2Loss, AddThreshold, SigmoidLoss, Sigmoid, TriangleInequality, \
    CoordinatesToDistanceMap, Condense, SpatialAttention, CoordinateScale, \
    NormalizedWeightedL2Loss
from pnet.utils.amino_acids import AminoAcid_SMILES
from pnet.models.conv_net_contact_map import ConvNetContactMapBase, to_one_hot, from_one_hot

class ConvNet3DStructureBase(ConvNetContactMapBase):
  """Base Class for Convolutional network contact map prediction"""
  def __init__(self,
               n_1D_feat=56,
               n_2D_feat=4,
               **kwargs):
    """
    Parameters:
    -----------
    n_1D_feat: int
      number of features for 1d inputs
    n_2D_feat: int
      number of features for 2d inputs
    uppertri: bool, optional
      build symmetry matrix(False) or upper-triangular matrix(True)
    """
    self.n_1D_feat = n_1D_feat
    self.n_2D_feat = n_2D_feat
    super(ConvNet3DStructureBase, self).__init__(**kwargs)
    
    
  def build_graph(self):
    """ Build graph structure """
    with self._get_tf("Graph").as_default():
      self.input_1D_features = Feature(shape=(self.batch_size, None, self.n_1D_feat), name='1D_features')
      self.input_2D_features = Feature(shape=(self.batch_size, None, None, self.n_2D_feat), name='2D_features')
      self.contact_prob = Feature(shape=(self.batch_size, None, None, 1), name='contact_prob')
      self.training_placeholder = Feature(shape=(), dtype=tf.bool, name='training_placeholder')
      # Placeholder for valid index
      self.res_flag_1D = Feature(shape=(self.batch_size, None), dtype=tf.int32, name='flag_1D')
      self.res_flag_2D = Feature(shape=(self.batch_size, None, None), dtype=tf.int32, name='flag_2D')
      self.n_residues = Feature(shape=(self.batch_size,), dtype=tf.int32, name='n_res')
      
      n_input = self.n_2D_feat + 1
      in_layer = Concat(axis=3, in_layers=[self.input_2D_features, self.contact_prob])

      # User-defined structures
      n_input, self.conv2d_out_layer = self.Conv2DModule(n_input, in_layer)
      n_input, self.condense_layer = self.CondenseModule(n_input, self.conv2d_out_layer)
      
      n_input += self.n_1D_feat
      in_layer = Concat(axis=2, in_layers=[self.condense_layer, self.input_1D_features])
      
      n_input, self.conv1d_out_layer = self.Conv1DModule(n_input, in_layer)

      n_out, self.cost = self.LossModule(n_input, self.conv1d_out_layer)
      self.set_loss(self.cost)      
      return 

  def LossModule(self, n_input, in_layer):
    
    distance_map = CoordinatesToDistanceMap(in_layers=[in_layer,
                                                       self.n_residues], name='distance_map')
    
    self.coordinate_labels = Label(shape=(None, 1), name='labels_coordinates')    
    self.coordinate_weights = Weights(shape=(None, 1), name='weights')
    cost = NormalizedWeightedL2Loss(in_layers=[distance_map, 
                                               self.coordinate_labels, 
                                               self.coordinate_weights], name='cost_r')
  
    self.add_output(distance_map)
    return 1, cost
  
  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True, 
                        **kwargs):
    """ Transform each batch into corresponding feed_dict """
    for epoch in range(epochs):
      for (X_b, twoD_X_b, y_b, w_b, oneD_y_b, oneD_w_b, contact_prob) in dataset.iterbatches(
          use_contact_prob=True,
          batch_size=self.batch_size,
          deterministic=True,
          pad_batches=pad_batches):
        
        n_residues = [seq_feat.shape[0] for seq_feat in X_b]
        max_n_res = max(n_residues)
        
        feed_dict = dict()
        if not self.training_placeholder.out_tensor is None:
          # This is always False, corresponding to instance normalization
          feed_dict[self.training_placeholder] = not predict
        
        if not oneD_y_b is None and not predict and not self.coordinate_labels.out_tensor is None:
          oneD_weights = []
          oneD_labels = []
          for oneD_label, oneD_weight in zip(oneD_y_b, oneD_w_b):
            raw_coordinates = np.pad(oneD_label, ((0, max_n_res-oneD_label.shape[0]), (0,0)), 'constant')
            dismap = np.expand_dims(raw_coordinates, 0) - np.expand_dims(raw_coordinates, 1)
            dismap = np.linalg.norm(dismap, ord=2, axis=-1)
            out_label = np.reshape(dismap, (-1, 1))
            
            sample_weight = np.pad(oneD_weight, ((0, max_n_res-oneD_weight.shape[0])), 'constant')
            weight_matrix = np.sign(np.expand_dims(sample_weight, 0)) * np.sign(np.expand_dims(sample_weight, 1))
            out_weight = np.reshape(weight_matrix, (-1, 1))
            
            oneD_labels.append((out_label*np.sign(out_weight))**2)
            oneD_weights.append(out_weight)
            
          feed_dict[self.coordinate_weights] = np.concatenate(oneD_weights, axis=0)
          feed_dict[self.coordinate_labels] = np.concatenate(oneD_labels, axis=0)
          
        res_features = []
        res_flag_1D = []
        res_flag_2D = []
        for ids, seq_feat in enumerate(X_b):
          n_res = n_residues[ids]
          # Padding
          flag_1D = [1]*n_res + [0]*(max_n_res-n_res)
          if self.uppertri:
            flag_2D = [[0]*k + flag_1D[k:] for k in range(n_res)] + [[0]*max_n_res]*(max_n_res-n_res)
          else:
            flag_2D = [flag_1D]*n_res + [[0]*max_n_res]*(max_n_res-n_res)
          res_flag_1D.append(np.array(flag_1D))
          res_flag_2D.append(np.array(flag_2D))
          res_features.append(np.pad(seq_feat, ((0, max_n_res - n_res), (0, 0)), 'constant'))
        
        res_2D_features = []
        for ids, twoD_feat in enumerate(twoD_X_b):
          n_res = n_residues[ids]
          # Padding
          res_2D_features.append(np.pad(twoD_feat, ((0, max_n_res - n_res), (0, max_n_res - n_res), (0, 0)), 'constant'))

        cont_probs = []
        for ids, cont_prob in enumerate(contact_prob):
          n_res = n_residues[ids]
          # Padding
          cont_probs.append(np.pad(cont_prob, ((0, max_n_res - n_res), (0, max_n_res - n_res)), 'constant'))
          
        if not self.input_1D_features.out_tensor is None:
          feed_dict[self.input_1D_features] = np.stack(res_features, axis=0)
        if not self.input_2D_features.out_tensor is None:
          feed_dict[self.input_2D_features] = np.stack(res_2D_features, axis=0)
        if not self.contact_prob.out_tensor is None:
          feed_dict[self.contact_prob] = np.expand_dims(np.stack(cont_probs, axis=0), axis=3)
        
        if not self.res_flag_1D.out_tensor is None:
          feed_dict[self.res_flag_1D] = np.stack(res_flag_1D, axis=0)
        if not self.res_flag_2D.out_tensor is None:
          feed_dict[self.res_flag_2D] = np.stack(res_flag_2D, axis=0)
        if not self.n_residues.out_tensor is None:
          feed_dict[self.n_residues] = np.array(n_residues)
        yield feed_dict

  def evaluate(self, dataset, metrics):
    """
    Evaluates the performance of this model on specified dataset.
    Parameters
    """
    # TODO: change this
    w_all = []
    y_all = []
    for _, _, y, w, _, _ in dataset.itersamples():
      w_all.append(np.sign(w))
      y_all.append(y)

    # Retrieve prediction label
    y_pred = self.predict_proba(dataset)
    if len(y_pred) == 2:
      y_pred = y_pred[1]
    # Mask all predictions and labels with valid index
    results = {}
    for metric in metrics:
      results[metric.name] = metric.compute_metric(y_all, y_pred, w=w_all)
    return results

  def predict_proba(self, dataset):
    gen = self.default_generator(dataset)
    y_pred = []
    for feed_dict in gen:
      y_pred.append(self.session.run(self.outputs[-1], feed_dict=feed_dict))
    return y_pred
  
class ConvNet3DStructure(ConvNet3DStructureBase):
  def __init__(self,
               **kwargs):
    super(ConvNet3DStructure, self).__init__(**kwargs)

  def Conv1DModule(self, n_input, in_layer):
    
    self.conv_1D_layers.append(Conv1DLayer(
        n_input_feat=n_input,
        n_output_feat=64,
        n_size=1,
        in_layers=[in_layer, self.res_flag_1D, self.training_placeholder], name='global_conv1d_0'))
    
    n_input = 64
    in_layer = self.conv_1D_layers[-1]
    
    for i in range(5):
      # n_input = 64
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, size=17, name='Res1D_LF_Module0_'+str(i)+'_')
    
    for i in range(5):
      # n_input = 64
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, name='Res1D_Module1_'+str(i)+'_')
    # n_input = 128
    #in_layer = SpatialAttention(in_layers=[in_layer, self.contact_prob, self.n_residues], name='spatial_attention')
    #n_input = n_input * 2
    
    for i in range(5):
      # n_input = 64
      n_input, in_layer = self.Res1DModule_b(n_input, in_layer, name='Res1D_Module2_'+str(i)+'_')

    unnormalized_coordinates = Dense(3, in_layers=[in_layer], name='generate_coordinates')
    
    output_coordinates = CoordinateScale(in_layers=[unnormalized_coordinates], name='output_scale')
    return 3, output_coordinates
    
  def Conv2DModule(self, n_input, in_layer):

    self.conv_2D_layers.append(Conv2DLayer(
        n_input_feat=n_input,
        n_output_feat=64,
        n_size=7,
        in_layers=[in_layer, self.res_flag_2D, self.training_placeholder], name='global_conv2d_0'))
    
    n_input = 64
    in_layer = self.conv_2D_layers[-1]
    
    for i in range(10):
      # n_input = 64
      n_input, in_layer = self.Res2DModule_b(n_input, in_layer, res_flag_2D=self.res_flag_2D, name='Res2D_Module'+str(i)+'_')
    # n_input = 128
    n_input, in_layer = self.Res2DModule_a(n_input, in_layer, res_flag_2D=self.res_flag_2D, name='Res2D_UpModule0_')
    # n_input = 256
    n_input, in_layer = self.Res2DModule_a(n_input, in_layer, res_flag_2D=self.res_flag_2D, name='Res2D_UpModule1_')
    return n_input, in_layer
  
  def CondenseModule(self, n_input, in_layer):
    out_layer = Condense(in_layers=[in_layer, self.contact_prob])
    # n_input = 512
    n_input = n_input# * 2
    return n_input, out_layer
