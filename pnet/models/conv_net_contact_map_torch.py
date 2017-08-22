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
from pnet.models.torch_base import TorchModel
from pnet.models.layers import TorchResidueEmbedding, \
    TorchOuter, TorchContactMapGather, TorchResAdd

class ConvNetContactMapTorch(TorchModel):

  def __init__(self,
               n_res_feat,
               embedding=True,
               embedding_length=50,
               filter_size_1D=[17]*6,
               n_filter_1D=[10]*3+[16]*3,
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
      whether to transfer the first 25 features(one hot encoding of residue
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
    super(ConvNetContactMapTorch, self).__init__(**kwargs)

  def build(self):
    """Constructs the graph architecture as specified in its config.
    """

    filter_size_1D = self.filter_size_1D
    n_filter_1D = self.n_filter_1D
    filter_size_2D = self.filter_size_2D
    n_filter_2D = self.n_filter_2D
    if self.embedding:
      self.residue_embedding = TorchResidueEmbedding(embedding_length=self.embedding_length)
      self.residue_embedding.cuda()
      self.n_res_feat = self.n_res_feat - 25 + self.embedding_length
    self.conv1 = torch.nn.Conv1d(self.n_res_feat, 10, 1)
    self.conv1.cuda()
    in_channels = 10

    i = 0
    self.conv1d_modules = []
    while i < len(n_filter_1D):
      self.conv1d_modules.append(torch.nn.Sequential(
          torch.nn.ReLU(),
          torch.nn.Conv1d(in_channels,
                          n_filter_1D[i],
                          filter_size_1D[i],
                          padding=(filter_size_1D[i]-1)//2),
          torch.nn.BatchNorm1d(n_filter_1D[i]),
          torch.nn.ReLU(),
          torch.nn.Conv1d(n_filter_1D[i],
                          n_filter_1D[i+1],filter_size_1D[i+1],
                          padding=(filter_size_1D[i+1]-1)//2),
          torch.nn.BatchNorm1d(n_filter_1D[i+1])))
      in_channels = n_filter_1D[i+1]
      i = i + 2
    for module in self.conv1d_modules:
      module.cuda()

    self.outer = TorchOuter()
    in_channels = 2 * in_channels

    i = 0
    self.conv2d_modules = []
    while i < len(n_filter_2D):
      self.conv2d_modules.append(torch.nn.Sequential(
          torch.nn.ReLU(),
          torch.nn.Conv2d(in_channels,
                          n_filter_2D[i],
                          filter_size_2D[i],
                          padding=(filter_size_2D[i]-1)//2),
          torch.nn.BatchNorm2d(n_filter_2D[i]),
          torch.nn.ReLU(),
          torch.nn.Conv2d(n_filter_2D[i],
                          n_filter_2D[i+1],
                          filter_size_2D[i+1],
                          padding=(filter_size_2D[i+1]-1)//2),
          torch.nn.BatchNorm2d(n_filter_2D[i+1])))
      in_channels = n_filter_2D[i+1]
      i = i + 2
    for module in self.conv2d_modules:
      module.cuda()

    self.gather = TorchContactMapGather(in_channels)
    self.gather.cuda()
    self.res_add = TorchResAdd()

    self.trainable_layers.extend(self.conv1d_modules)
    self.trainable_layers.extend(self.conv2d_modules)
    self.trainable_layers.extend([self.residue_embedding, self.conv1, self.gather])

  def forward(self, inputs, training=False):
    X_batch = inputs[0]
    print("batch of size" + str(X_batch.shape))
    X = torch.autograd.Variable(torch.cuda.FloatTensor(X_batch))
    flags_batch = inputs[1]
    flags = torch.autograd.Variable(torch.cuda.FloatTensor(flags_batch))
    if self.embedding:
      out = self.residue_embedding(X)
    else:
      out = X
    out = self.conv1(out)
    for module in self.conv1d_modules:
      conv_out = module(out)
      out = self.res_add(conv_out, out)
    out = self.outer(out, flags=flags)
    for module in self.conv2d_modules:
      conv_out = module(out)
      out = self.res_add(conv_out, out)
    y_pred = self.gather(out, flags=flags)
    if not training:
      y_pred = torch.nn.Softmax(y_pred)
    return [y_pred]

  def cost(self, logit, label, weight):
    label = torch.autograd.Variable(torch.cuda.FloatTensor(label))
    weight = torch.autograd.Variable(torch.cuda.FloatTensor(weight))
    loss_func = torch.nn.CrossEntropyLoss(size_average=False)
    loss = []
    for i in range(logit.size()[0]):
      loss.append(loss_func(logit[i:i+1, :], label[i:i+1].long()).mul(weight[i:i+1].float()))
    loss = torch.cat(loss).sum()
    return loss

  def predict_on_batch(self, inputs):
    y_pred_batch = self.forward(inputs, training=False)
    y_pred_batch = y_pred_batch.data.cpu().numpy()[:]
    return y_pred_batch

  def predict_proba_on_batch(self, X_batch):
    return self.predict_on_batch(X_batch)


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
        if not y_b is None and not predict:
          labels = []
          for ids, label in enumerate(y_b):
            labels.append(label.flatten())
          y_out = np.reshape(np.concatenate(labels, axis=0), [-1, 1]).astype(float)

        if not w_b is None and not predict:
          weights = []
          for ids, weight in enumerate(w_b):
            weights.append(weight.flatten())
          w_out = np.reshape(np.concatenate(weights, axis=0), [-1, 1])

        res_features = []
        res_flag_2D = []
        n_residues = [seq_feat.shape[0] for seq_feat in X_b]
        max_n_res = max(n_residues)
        for ids, seq_feat in enumerate(X_b):
          n_res = n_residues[ids]
          # Padding
          flag_1D = [1]*n_res + [0]*(max_n_res-n_res)
          flag_2D = [flag_1D]*n_res + [[0]*max_n_res]*(max_n_res-n_res)
          res_flag_2D.append(np.array(flag_2D))
          res_features.append(np.transpose(np.pad(seq_feat, ((0, max_n_res - n_res), (0, 0)), 'constant')))

        X_out = np.stack(res_features, axis=0)
        flags_out = np.stack(res_flag_2D, axis=0)
        yield [X_out, flags_out], y_out, w_out

  def evaluate(self, dataset, metrics):
    """
    Evaluates the performance of this model on specified dataset.
    Parameters
    """
    w_all = []
    y_all = []
    for X, y, w in dataset.itersamples():
      w_sample = np.sign(w)
      n_residues = w_sample.shape[0]
      full_range = np.abs(np.stack([np.arange(n_residues)] * n_residues, axis=0) -
                   np.stack([np.arange(n_residues)] * n_residues, axis=1))
      range_short = ((11 - full_range ) >= 0).astype(float) * ((full_range - 6 ) >= 0).astype(float) * w_sample
      range_medium = ((23 - full_range ) >= 0).astype(float) * ((full_range - 12 ) >= 0).astype(float)  * w_sample
      range_long = ((full_range - 24 ) >= 0).astype(float) * w_sample
      w_all.append(np.stack([range_short.flatten(), range_medium.flatten(), range_long.flatten()], 1))
      y_all.append(y.flatten())

    w = np.concatenate(w_all, axis=0)
    # Retrieve prediction and true label
    y_pred = self.predict_proba(dataset)
    y = np.concatenate(y_all)*1
    # Mask all predictions and labels with valid index
    results = [{}, {}, {}]
    for i in range(3):
      y_eval = y_pred[np.nonzero(w[:, i])]
      y_true = y[np.nonzero(w[:, i])]
      # Calculate performances
      for metric in metrics:
        results[i][metric.name] = metric.compute_metric(y_true, y_eval)
    return results