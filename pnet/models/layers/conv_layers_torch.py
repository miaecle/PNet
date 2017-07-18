#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:57:45 2017

@author: zqwu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import torch

def get_fans(shape):
  if len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  elif len(shape) == 4 or len(shape) == 5:
    # Assuming convolution kernels (2D or 3D).
    # TF kernel shape: (..., input_depth, depth)
    receptive_field_size = np.prod(shape[:2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  else:
    # No specific assumptions.
    fan_in = np.sqrt(np.prod(shape))
    fan_out = np.sqrt(np.prod(shape))
  return fan_in, fan_out

def initialization(shape, mode='glorot_normal', gpu=True, requires_grad=True):

  if mode == 'glorot_normal':
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    init = np.random.normal(0, s, shape)
  elif mode == 'glorot_uniform':
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    init = np.random.uniform(-s, s, shape)
  elif mode == 'zeros':
    init = np.zeros(shape)
  elif mode == 'ones':
    init = np.ones(shape)
  else:
    raise ValueError('Not supported mode')
  if gpu:
    init = torch.cuda.FloatTensor(init)
  if requires_grad:
    return torch.nn.Parameter(init)
  else:
    return torch.autograd.Variable(init, requires_grad=requires_grad)


class TorchResidueEmbedding(torch.nn.Module):

  def __init__(self,
               pos_start=0,
               pos_end=23,
               embedding_length=50,
               **kwargs):
    """
    Parameters
    ----------
    pos_start: int, optional
      starting position of raw features that need embedding
    pos_end: int, optional
      ending position
    embedding_length: int, optional
      length for embedding
    """
    super(TorchResidueEmbedding, self).__init__(**kwargs)
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.embedding_length = embedding_length
    self.embedding = initialization((self.pos_end - self.pos_start, self.embedding_length))

  def forward(self, x):
    res_one_hot = torch.transpose(x[:, self.pos_start:self.pos_end, :], 1, 2)
    embedded_features = torch.transpose(torch.matmul(res_one_hot, self.embedding), 1, 2)
    out_tensor = torch.cat([embedded_features, x[:, self.pos_end:, :]], 1)
    return out_tensor

class TorchOuter(torch.nn.Module):
  def forward(self, X, flags=None):
    """ parent layers: input_features, input_flag_2D
    This layer concats 1D sequences into 2D sequences
    """
    seq_length = X.size()[-1]
    tensor1 = torch.unsqueeze(X, 3)
    tensor1 = torch.cat([tensor1]*seq_length, 3)
    tensor2 = torch.unsqueeze(X, 2)
    tensor2 = torch.cat([tensor2]*seq_length, 2)
    out_tensor = torch.cat([tensor1, tensor2], 1)
    if not flags is None:
      flags = torch.unsqueeze(flags, 1)
      out_tensor = out_tensor * flags
    return out_tensor

class TorchContactMapGather(torch.nn.Module):

  def __init__(self,
               n_res_feat,
               **kwargs):
    super(TorchContactMapGather, self).__init__(**kwargs)
    self.n_res_feat = n_res_feat
    self.linear = torch.nn.Linear(n_res_feat, 2)
    self.relu = torch.nn.ReLU()

  def forward(self, X, flags=None):
    """ parent layers: input_features, input_flag_2D
    """
    # Rearrange as Channel, Batch, W, H
    input_features = torch.transpose(X, 0, 1).contiguous()
    n_feat = input_features.size()[0]
    out_tensor = input_features.view([n_feat, -1])

    if not flags is None:
      flags = torch.unsqueeze(flags.contiguous().view([-1]), 0).byte()
      out_tensor = out_tensor.masked_select(flags).view([n_feat, -1])

    out_tensor = self.linear(out_tensor.t())
    out_tensor = self.relu(out_tensor)
    return out_tensor

class TorchResAdd(torch.nn.Module):
  def forward(self, fx, x):
    """ parent layers: fx, x
    """
    fx_in_channels = fx.size()[1]
    x_in_channels = x.size()[1]
    pad_size = list(x.size())
    pad_length = fx_in_channels - x_in_channels
    pad_size[1] = pad_length
    assert pad_length > 0

    out_tensor = fx + torch.cat([x, torch.zeros(pad_size)], 1)
    return out_tensor