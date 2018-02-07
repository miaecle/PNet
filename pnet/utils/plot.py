#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 00:20:36 2018

@author: zqwu
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

def plot_sample(model, sample, path):
  y_pred = model.predict_proba(sample)
  i = 0
  j = 0
  for batch in sample.itersamples():
    y = batch[2]
    ID = sample._IDs[i]
    plot(y, os.path.join(path, ID + '_y_true.png'), title=ID+"_RR_contact")
    w = batch[3]
    plot(w, os.path.join(path, ID + '_w.png'), title=ID+"_RR_contact_weights")
    n_res = y.shape[0]
    y_pred_sample = y_pred[j:(j+n_res*(n_res+1)/2), 1]
    y_pred_out = np.zeros((n_res, n_res))
    k = 0
    for ii in range(n_res):
      y_pred_out[ii, ii:] = y_pred_sample[k:k+(n_res-ii)]
      k += (n_res-ii)
    y_pred_out = y_pred_out + np.transpose(y_pred_out)
    plot(y_pred_out, os.path.join(path, ID + '_y_pred.png'), title=ID+"_RR_contact_predictions")
    j += n_res*(n_res+1)/2
    i += 1

def plot(matrix, path, title="RR_contact"):
  plt.clf()
  fig, ax = plt.subplots()
  if len(np.unique(matrix)) == 2:
    binary = True
  else:
    binary = False
  if binary:
    cmap = matplotlib.cm.get_cmap('binary')
  else:
    cmap = matplotlib.cm.get_cmap('viridis')
    cmap.set_under('w')
  cax = ax.imshow(matrix, cmap=cmap, vmin=.001)
  ax.set_title(title)
  ax.set_xlabel("Amino Acid Index")
  ax.set_ylabel("Amino Acid Index")
  if not binary:
    fig.colorbar(cax, cmap = matplotlib.cm.get_cmap(cmap))
  plt.savefig(path, dpi=300)
  return