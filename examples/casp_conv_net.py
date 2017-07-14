#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:43:49 2017

@author: zqwu
"""
import deepchem as dc
import numpy as np
import pnet
import os

datasets = pnet.utils.load_CASP_all()
train, valid, test = datasets.train_valid_test_split(deterministic=True)
data_dir = os.path.join(os.environ['PNET_DATA_DIR'], 'CASPALL')
name_datasets = ['train', 'valid', 'test']

for i, dataset in enumerate([train, valid, test]):
  path = os.path.join(data_dir, name_datasets[i])
  dataset.build_features(['raw', 'MSA', 'SS', 'SA'], path=path)
  dataset.build_labels(path=path)

batch_size = 4
n_features = train.n_features
metrics = [dc.metrics.Metric(dc.metrics.accuracy_score, np.mean, mode="classification")]

model = pnet.models.ConvNetContactMap(
    n_res_feat=n_features,
    learning_rate=1e-3,
    batch_size=batch_size,
    use_queue=False,
    mode='classification')

model.fit(train, nb_epoch=5, checkpoint_interval=20)
valid_scores = model.evaluate(valid, metrics)
print(valid_scores)
model.fit(train, nb_epoch=5)
valid_scores = model.evaluate(valid, metrics)
print(valid_scores)
model.fit(train, nb_epoch=5)
valid_scores = model.evaluate(valid, metrics)
print(valid_scores)
model.fit(train, nb_epoch=5)
valid_scores = model.evaluate(valid, metrics)
print(valid_scores)

train_scores = model.evaluate(train, metrics)
print(train_scores)

valid_scores = model.evaluate(valid, metrics)
print(valid_scores)
