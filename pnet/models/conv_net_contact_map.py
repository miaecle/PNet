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
from pnet.models.layers import ResidueEmbedding, Conv1DLayer, Conv2DLayer, Outer1DTo2DLayer, ContactMapGather

def to_one_hot(y, n_classes=2):
  """Transforms label vector into one-hot encoding.

  Turns y into vector of shape [n_samples, 2] (assuming binary labels).

  y: np.ndarray
    A vector of shape [n_samples, 1]
  """
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, n_classes))
  y_hot[np.arange(n_samples), y.astype(np.int64)] = 1
  return y_hot


def from_one_hot(y, axis=1):
  """Transorms label vector from one-hot encoding.

  y: np.ndarray
    A vector of shape [n_samples, num_classes]
  """
  return np.argmax(y, axis=axis)

class ConvNetContactMap(TensorGraph):
  def __init__(self,
               n_res_feat,
               batch_size,
               embedding=True,
               embedding_length=50,
               filter_size_1D=[51, 25, 11],
               n_filter_1D=[50, 50, 50],
               filter_size_2D=[25, 25, 25],
               n_filter_2D=[50, 50, 50],
               max_n_res=1000,
               **kwargs):
    self.n_res_feat = n_res_feat
    self.batch_size = batch_size
    self.embedding = embedding
    self.embedding_length = embedding_length
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
    #self.n_residues = Feature(shape=(self.batch_size), dtype=tf.int32)
    self.conv_1D_layers = []
    self.batch_norm_layers = []
    n_input = self.n_res_feat
    in_layer = self.res_features
    if self.embedding:
      self.residues_embedding = ResidueEmbedding(
          pos_start=0,
          pos_end=23,
          embedding_length=self.embedding_length,
          in_layers=[in_layer])
      n_input = n_input - 23 + self.embedding_length
      in_layer = self.residues_embedding
    for i, layer_1D in enumerate(self.n_filter_1D):
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
    for i, layer_2D in enumerate(self.n_filter_2D):
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

    self.contact_labels = Label(shape=(None, 2))
    self.contact_weights = Weights(shape=(None,))
    cost = SoftMaxCrossEntropy(in_layers=[self.contact_labels, self.gather_layer])
    all_loss = WeightedError(in_layers=[cost, self.contact_weights])
    self.set_loss(all_loss)
    return

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True):

    for epoch in range(epochs):
      for (X_b, y_b, w_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=True,
          pad_batches=pad_batches):

        feed_dict = dict()
        if not y_b is None and not predict:
          labels = []
          for ids, label in enumerate(y_b):
            labels.append(label.flatten())
          feed_dict[self.contact_labels] = to_one_hot(np.concatenate(labels, axis=0))

        if not w_b is None and not predict:
          weights = []
          for ids, weight in enumerate(w_b):
            weights.append(weight.flatten())
          feed_dict[self.contact_weights] = np.concatenate(labels, axis=0)

        res_features = []
        res_flag_1D = []
        res_flag_2D = []
        n_residues = []
        for ids, seq_feat in enumerate(X_b):
          n_res, n_features = seq_feat.shape
          assert n_features == self.n_res_feat
          flag_1D = [1]*n_res + [0]*(self.max_n_res-n_res)
          flag_2D = [flag_1D]*n_res + [[0]*self.max_n_res]*(self.max_n_res-n_res)
          n_residues.append(n_res)
          res_flag_1D.append(np.array(flag_1D))
          res_flag_2D.append(np.array(flag_2D))
          res_features.append(np.pad(seq_feat, ((0, self.max_n_res - n_res), (0, 0)), 'constant'))

        feed_dict[self.res_features] = np.stack(res_features, axis=0)
        feed_dict[self.res_flag_1D] = np.stack(res_flag_1D, axis=0)
        feed_dict[self.res_flag_2D] = np.stack(res_flag_2D, axis=0)
        #feed_dict[self.n_residues] = np.array(n_residues)
        yield feed_dict