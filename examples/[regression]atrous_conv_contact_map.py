#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:14:44 2018

@author: zqwu
"""

import deepchem as dc
import numpy as np
import pnet
import os

train = pnet.utils.load_PDB50_selected()
data_dir_train = os.path.join(os.environ['PNET_DATA_DIR'], 'PDB50selected')
train.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_train)
# Regression settings
train.build_labels(path=data_dir_train, weight_base=2.7, weight_adjust=0.001, binary=False)

CASPALL = pnet.utils.load_CASP_all()
data_dir_valid = os.path.join(os.environ['PNET_DATA_DIR'], 'CASPALL')
CASPALL.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_valid)
CASPALL.build_labels(path=data_dir_valid, weight_base=50., weight_adjust=0.1, binary=True)

batch_size = 1
n_features = CASPALL.n_features
metrics = [pnet.utils.Metric(pnet.utils.top_k_accuracy(5, input_mode='regression'))]
model_dir = '/home/zqwu/PNet/built_models/AtrousConv_PDB50selected_reg'

model = pnet.models.AtrousConvContactMap(
    n_res_feat=n_features,
    learning_rate=1e-4,
    learning_rate_decay=0.95,
    batch_size=batch_size,
    use_queue=False,
    uppertri=True,
    mode='regression',
    n_batches=None,
    model_dir=model_dir)

model.build()
#model.restore()

model.fit(train, nb_epoch=25, checkpoint_interval=11498)


CASP11 = pnet.utils.load_CASP(11)
CASP11 = CASPALL.select_by_ID(CASP11._IDs)
CASP12 = pnet.utils.load_CASP(12)
CASP12 = CASPALL.select_by_ID(CASP12._IDs)
print(model.evaluate(CASPALL, metrics))
print(model.evaluate(CASP11, metrics))
print(model.evaluate(CASP12, metrics))

metrics2 = [pnet.utils.Metric(pnet.utils.top_k_accuracy(5, input_mode='regression'))]
print(model.evaluate(CASP11, metrics2))
print(model.evaluate(CASP12, metrics2))