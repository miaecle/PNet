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
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.embedding_length = embedding_length
    super(TorchResidueEmbedding, self).__init__(**kwargs)

  def build(self):
    """ Construct internal trainable weights.
    """
    
    self.embedding = initialization((self.pos_end - self.pos_start, self.embedding_length))

  def forward(self, x):
    '''
    self.build()
    embedded_features = tf.tensordot(x[:, self.pos_start:self.pos_end, :], self.embedding, [[2], [0]])
    out_tensor = tf.concat([embedded_features, input_features[:, :, self.pos_end:]], axis=2)
    if set_tensors:
      self.variables = self.trainable_weights
      self.out_tensor = out_tensor
    return out_tensor
    '''