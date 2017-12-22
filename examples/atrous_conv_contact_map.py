cs#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:43:49 2017

@author: zqwu
"""
import deepchem as dc
import numpy as np
import pnet
import os
from sklearn.metrics import r2_score
"""
train = pnet.utils.load_PDB50_selected()
data_dir_train = os.path.join(os.environ['PNET_DATA_DIR'], 'PDB50selected')
train.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_train)
train.build_labels(path=data_dir_train, weight_adjust=30., binary=True)
"""
CASPALL = pnet.utils.load_CASP_all()
data_dir_valid = os.path.join(os.environ['PNET_DATA_DIR'], 'CASPALL')
CASPALL.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_valid)
CASPALL.build_labels(path=data_dir_valid, weight_adjust=30., binary=True)

train, valid, test = CASPALL.train_valid_test_split()

batch_size = 1
n_features = valid.n_features
#metrics = [pnet.utils.Metric(pnet.utils.top_k_accuracy(5), mode='classification')]
model_dir = '/home/zqwu/PNet/built_models/AtrousConv_CASPALL'

model = pnet.models.AtrousConvContactMap(
    n_res_feat=n_features,
    learning_rate=1e-4,
    learning_rate_decay=0.95,
    batch_size=batch_size,
    use_queue=False,
    uppertri=True,
    mode='classification',
    model_dir=model_dir)

model.build()
#model.restore()

model.fit(train, nb_epoch=1000, checkpoint_interval=100)

def evaluate(model, dataset):
  gen = model.default_generator(dataset)
  y_preds = []
  scores = []
  for feed_dict in gen:
    w = feed_dict[model.oneD_weights]
    y_pred = model.session.run(model.oneD_prediction.out_tensor, feed_dict=feed_dict)
    y_preds.append(y_pred)
    y_true = feed_dict[model.oneD_labels]
    if sum(w) > 0:    
      y_t = y_true[np.where(w != 0)[0], :]
      y_p = y_pred[np.where(w != 0)[0], :]
      scores.append([r2_score(y_t[:, i], y_p[:, i])for i in range(3)])
  return np.average(np.array(scores), axis=0)

print(evaluate(model, valid))
print(evaluate(model, test))