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

train = pnet.utils.load_PDB50_selected()
data_dir_train = os.path.join(os.environ['PNET_DATA_DIR'], 'PDB50selected')
train.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_train)
train.build_labels(path=data_dir_train, weight_adjust=30., binary=True)

valid = pnet.utils.load_CASP_all()
data_dir_valid = os.path.join(os.environ['PNET_DATA_DIR'], 'CASPALL')
valid.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_valid)
valid.build_labels(path=data_dir_valid, weight_adjust=30., binary=True)

batch_size = 1
n_features = train.n_features
metrics = [pnet.utils.Metric(pnet.utils.top_k_accuracy(5), mode='classification')]
model_dir = '/home/zqwu/PNet/built_models/AtrousConv_PDB50selected'

model = pnet.models.AtrousConvContactMap(
    n_res_feat=n_features,
    learning_rate=1e-5,
    batch_size=batch_size,
    use_queue=False,
    uppertri=True,
    mode='regression',
    model_dir=model_dir)

#model.restore()

model.fit(train, nb_epoch=10, checkpoint_interval=500)

train_scores = model.evaluate(train, metrics)
print(train_scores)

valid_scores = model.evaluate(valid, metrics)
print(valid_scores)
