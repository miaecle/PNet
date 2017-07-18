#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:59:10 2017

@author: zqwu
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:31:24 2017

@author: Zhenqin Wu
"""

import torch
import numpy as np
import pnet
from pnet.models.torch_base import TorchModel


class ConvNetContactMapTorch(TorchModel):

  def __init__(self,
               n_res_feat,
               embedding=True,
               embedding_length=50,
               filter_size_1D=[17]*6,
               n_filter_1D=[6]*6,
               filter_size_2D=[3]*10,
               n_filter_2D=list(range(35, 65, 5))+[60]*4,
               max_n_res=1000,
               **kwargs):
    """
    Parameters:
    -----------
    n_res_feat: int
      number of features for each residue
    embedding: bool, optional
      whether to transfer the first 23 features(one hot encoding of residue
      type) to variable embedding
    embedding_length: int, optional
      length of embedding
    filter_size_1D: list, optional
      structure of 1D convolution: size of convolution
    n_filter_1D: list, optional
      structure of 1D convolution: depths of convolution
    filter_size_2D: list, optional
      structure of 2D convolution: size of convolution
    n_filter_2D: list, optional
      structure of 2D convolution: depths of convolution
    max_n_res: int, optional
      maximum number of residues, used for padding
    """
    self.n_res_feat = n_res_feat
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

  def build(self):
    """Constructs the graph architecture as specified in its config.
    """

    filter_size_1D = self.filter_size_1D
    n_filter_1D = self.n_filter_1D
    filter_size_2D = self.filter_size_2D
    n_filter_2D = self.n_filter_2D
    self.conv_1d_layers = []
    self.conv_2d_layers = []
    self.batch_norm_layers = []
    self.residue_embedding = TorchResidueEmbedding()

    self.conv_layers.append(torch.nn.Conv1d(self.n_res_feat, n_filter_1D[0], filter_size_1D[0]))
    in_channels = n_filter_1D[0]
    for i, filter_1D in enumerate(filter_size_1D):
      self.conv_layers.append(torch.nn.Conv1d(in_channels, n_filter_1D[i], filter_1D))
      in_channels = n_filter_1D[i]
      self.batch_norm_layers.append(torch.nn.BatchNorm1d(in_channels))

    in_channels = 2 * in_channels
    for i, filter_2D in enumerate(filter_size_2D):
      self.conv_2d_layers.append(torch.nn.Conv2d(in_channels, n_filter_2D[i], filter_2D))
      in_channels = n_filter_2D[i]
      self.batch_norm_layers.append(torch.nn.BatchNorm2d(in_channels))

    self.gather_layer = TorchContactMapGather()

  def forward(self, X, training=False):
    X = self.residue_embedding(X)

    return outputs

  def cost(self, logit, label, weight):
    '''
    loss = []
    loss_func = torch.nn.MSELoss()
    for i in range(logit.size()[0]):
      loss.append(loss_func(logit[i], label[i]).mul(weight[i]))
    loss = torch.cat(loss).mean()
    '''
    return loss

  def predict_on_batch(self, X_batch):
    '''
    X_batch = torch.autograd.Variable(torch.cuda.FloatTensor(X_batch))
    outputs = self.forward(X_batch, training=False)
    y_pred_batch = torch.stack(outputs, 1).data.cpu().numpy()[:]
    y_pred_batch = np.squeeze(y_pred_batch, axis=2)
    '''
    return y_pred_batch

  def predict_proba_on_batch(self, X_batch):
    return predict_on_batch(X_batch)


  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        pad_batches=True):