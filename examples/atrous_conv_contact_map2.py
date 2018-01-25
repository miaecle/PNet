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
train.build_labels(path=data_dir_train, weight_base=50., weight_adjust=0.1, binary=True)

CASPALL = pnet.utils.load_CASP_all()
data_dir_valid = os.path.join(os.environ['PNET_DATA_DIR'], 'CASPALL')
CASPALL.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_valid)
CASPALL.build_labels(path=data_dir_valid, weight_base=50., weight_adjust=0.1, binary=True)


batch_size = 1
n_features = CASPALL.n_features
metrics = [pnet.utils.Metric(pnet.utils.top_k_accuracy(5), mode='classification')]
model_dir = '/home/zqwu/PNet/built_models/AtrousConv_PDB50selected'

model = pnet.models.AtrousConvContactMap(
    n_res_feat=n_features,
    learning_rate=1e-3,
    learning_rate_decay=0.95,
    batch_size=batch_size,
    use_queue=False,
    uppertri=True,
    mode='classification',
    n_batches=None,
    model_dir=model_dir)

model.restore()
model.fit(train, nb_epoch=5, checkpoint_interval=11498)
CASP11 = pnet.utils.load_CASP(11)
CASP11 = CASPALL.select_by_ID(CASP11._IDs)
CASP12 = pnet.utils.load_CASP(12)
CASP12 = CASPALL.select_by_ID(CASP12._IDs)
print(model_.evaluate(CASPALL, metrics))
print(model_.evaluate(CASP11, metrics))
print(model_.evaluate(CASP12, metrics))

metrics2 = [pnet.utils.Metric(pnet.utils.top_k_accuracy(10), mode='classification')]
print(model_.evaluate(CASP11, metrics2))
print(model_.evaluate(CASP12, metrics2))

model.fit(train, nb_epoch=5, checkpoint_interval=11498)
CASP11 = pnet.utils.load_CASP(11)
CASP11 = CASPALL.select_by_ID(CASP11._IDs)
CASP12 = pnet.utils.load_CASP(12)
CASP12 = CASPALL.select_by_ID(CASP12._IDs)
print(model_.evaluate(CASPALL, metrics))
print(model_.evaluate(CASP11, metrics))
print(model_.evaluate(CASP12, metrics))

metrics2 = [pnet.utils.Metric(pnet.utils.top_k_accuracy(10), mode='classification')]
print(model_.evaluate(CASP11, metrics2))
print(model_.evaluate(CASP12, metrics2))

