#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:43:49 2017

@author: zqwu
"""
import deepchem as dc
import numpy as np
import pnet

dataset = pnet.utils.load_CASP_all()
train, valid, test = dataset.train_valid_test_split(deterministic=True)

for i in [train, valid, test]:
  i.build_features(['raw', 'MSA', 'SS', 'SA'])
  i.build_labels()

n_samples = dataset.n_samples
batch_size = 4
n_features = train.n_features
metrics = [dc.metrics.Metric(dc.metrics.accuracy_score, np.mean, mode="classification")]

model = pnet.models.ConvNetContactMap(
    n_res_feat=n_features,
    learning_rate=1e-3,
    batch_size=batch_size,
    use_queue=False,
    mode='classification')

model.fit(train)

train_scores = model.evaluate(train, metrics)
print(train_scores)

valid_scores = model.evaluate(valid, metrics)
print(valid_scores)
