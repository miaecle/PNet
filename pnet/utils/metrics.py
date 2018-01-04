#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:07:22 2017

@author: zqwu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

class Metric(deepchem.metrics.Metric):

  def __init__(self,
               metric,
               name=None,
               mode=None):
    """
    Args:
      metric: customized function that takes args y_true, y_pred, w and
              computes desired score.
    """
    self.metric = metric
    if name is None:
      self.name = self.metric.__name__
    else:
      self.name = name
    if mode is None:
      mode = "classification"
    assert mode in ["classification", "regression"]
    self.mode = mode

  @staticmethod
  def uppertri(y):
    assert y.shape[0] == y.shape[1]
    assert len(y.shape) >= 2
    return np.concatenate([y[k, k:] for k in range(y.shape[0])], axis=0)
  
  def compute_metric(self,
                     y_true,
                     y_pred,
                     w=None,
                     n_classes=2,
                     separate_range=True):
    """Compute a performance metric for each task.

    Parameters
    ----------
    y_true: list of np.ndarray
      An np.ndarray containing true values for each task.
    y_pred: np.ndarray
      An np.ndarray containing predicted values for each task.
    w: list of np.ndarray, optional
      An np.ndarray containing weights for each datapoint.
    n_classes: int, optional
      Number of classes in data for classification tasks.
    separate_range: bool, optional
      Whether to divide the results into three different distance ranges

    Returns
    -------
    A numpy nd.array containing metric values
    """
    if w is None or len(w) == 0:
      w = [np.ones_like(y_sample) for y_sample in y_true]
    _y = []
    _w = []
    for i, y_sample in enumerate(y_true):
      sample = self.uppertri(y_sample)
      _y.append(np.stack([sample, np.ones_like(sample)*i], 1))

      if separate_range:
        n_residues = w[i].shape[0]
        full_range = np.abs(np.stack([np.arange(n_residues)] * n_residues,
                                      axis=0) - \
            np.stack([np.arange(n_residues)] * n_residues, axis=1))
        range_short = ((11 - full_range ) >= 0).astype(float) * \
            ((full_range - 6 ) >= 0).astype(float) * w[i]
        range_medium = ((23 - full_range ) >= 0).astype(float) * \
            ((full_range - 12 ) >= 0).astype(float)  * w[i]
        range_long = ((full_range - 24 ) >= 0).astype(float) * w[i]
        _w.append(np.stack([self.uppertri(range_short),
                            self.uppertri(range_medium),
                            self.uppertri(range_long)], 1))
      else:
        _w.append(self.uppertri(w[i]))
    _y = np.concatenate(_y, 0)
    _w = np.concatenate(_w, 0)
    assert _y.shape[0] == y_pred.shape[0] == _w.shape[0]

    metric_value = self.metric(_y, y_pred, _w)
    print("computed_metrics: %s" % str(metric_value))
    return metric_value

def from_one_hot(y, axis=1):
  """Transorms label vector from one-hot encoding.

  y: np.ndarray
    A vector of shape [n_samples, num_classes]
  """
  return np.argmax(y, axis=axis)

def pnet_roc_auc_score(y, y_pred, w):
  scores = []
  for i in range(w.shape[1]):
    y_eval = y_pred[np.nonzero(w[:, i])]
    y_true = np.squeeze(y[np.nonzero(w[:, i]), 0])
    scores.append(roc_auc_score(y_true, y_eval[:, 1]))
  return scores

def pnet_recall_score(y, y_pred, w):
  scores = []
  for i in range(w.shape[1]):
    y_eval = y_pred[np.nonzero(w[:, i])]
    y_true = np.squeeze(y[np.nonzero(w[:, i]), 0])
    scores.append(recall_score(y_true, from_one_hot(y_eval)))
  return scores

def pnet_accuracy_score(y, y_pred, w):
  scores = []
  for i in range(w.shape[1]):
    y_eval = y_pred[np.nonzero(w[:, i])]
    y_true = np.squeeze(y[np.nonzero(w[:, i]), 0])
    scores.append(accuracy_score(y_true, from_one_hot(y_eval)))
  return scores

def pnet_precision_score(y, y_pred, w):
  scores = []
  for i in range(w.shape[1]):
    y_eval = y_pred[np.nonzero(w[:, i])]
    y_true = np.squeeze(y[np.nonzero(w[:, i]), 0])
    scores.append(precision_score(y_true, from_one_hot(y_eval)))
  return scores

def pnet_prc_auc_score(y, y_pred, w):
  scores = []
  for i in range(w.shape[1]):
    y_eval = y_pred[np.nonzero(w[:, i])]
    y_true = np.squeeze(y[np.nonzero(w[:, i]), 0])
    precision, recall, _ = precision_recall_curve(y_true, y_eval[:, 1])
    scores.append(auc(recall, precision))
  return scores

class top_k_accuracy(object):
  """ Accuracy of the top L/k predictions, L is the length of protein
  """
  def __init__(self, k=5):
    self.k = k
    self.__name__ = 'top_k_accuracy'

  def __call__(self, y, y_pred, w):
    scores = []
    # Partition of proteins
    _, partition_index = np.unique(y[:, 1], return_index=True)
    partition_index = list(partition_index)
    partition_index.append(len(y[:, 1]))
    for j in range(w.shape[1]):
      # Masking of different ranges
      y_eval = y_pred[:, 1] * np.sign(w[:, j])
      # Divide into separate partitions
      y_partition = [y_eval[partition_index[i]:partition_index[i+1]] for i in range(len(partition_index)-1)]
      out = []
      for sample in y_partition:
        n_residues = np.floor(np.sqrt(len(sample)*2))
        # Number of predictions in evaluation
        n_eval = (n_residues/self.k).astype(int)
        out.append(np.greater(sample, sorted(sample)[-n_eval-1]) * 1)
      out = np.concatenate(out, 0)
      scores.append(sum(out * y[:, 0])/min(sum(out), sum(y[:,0]*w[:, j])))
    return scores
