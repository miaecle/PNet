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

datasets = pnet.utils.load_PDB50()
data_dir = os.path.join(os.environ['PNET_DATA_DIR'], 'PDB50ALL')
datasets.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir)
datasets.build_labels(path=data_dir, weight_adjust=30.)

train, valid, test = datasets.train_valid_test_split(deterministic=True)

batch_size = 1
n_features = train.n_features
metrics = [dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")]

model = pnet.models.ConvNetContactMap(
    n_res_feat=n_features,
    learning_rate=1e-3,
    batch_size=batch_size,
    use_queue=False,
    mode='classification')

model.fit(train, nb_epoch=200, checkpoint_interval=20)

train_scores = model.evaluate(train, metrics)
print(train_scores)

valid_scores = model.evaluate(valid, metrics)
print(valid_scores)
