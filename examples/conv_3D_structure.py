#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:34:05 2018

@author: zqwu
"""

import deepchem as dc
import numpy as np
import pnet
import os
import joblib
import tensorflow as tf

train = pnet.utils.load_PDB50_selected()
data_dir_train = os.path.join(os.environ['PNET_DATA_DIR'], 'PDB50selected')
train.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_train)
train.build_labels(path=data_dir_train, weight_base=50., weight_adjust=0.1, binary=True)
train.build_contact_prob(data_dir_train)

CASPALL = pnet.utils.load_CASP_all()
data_dir_valid = os.path.join(os.environ['PNET_DATA_DIR'], 'CASPALL')
CASPALL.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_valid)
CASPALL.build_labels(path=data_dir_valid, weight_base=50., weight_adjust=0.1, binary=True)
CASPALL.build_contact_prob(data_dir_valid)

batch_size = 1
model_dir = '/home/zqwu/PNet/built_models/Conv3DStructure_PDB50selected'

model = pnet.models.ConvNet3DStructure(
    n_1D_feat=56,
    n_2D_feat=4,
    learning_rate=1e-5,
    learning_rate_decay=0.99,
    batch_size=batch_size,
    use_queue=False,
    uppertri=True,
    mode='classification',
    n_batches=None,
    oneD_loss=None,
    model_dir=model_dir)


model.build()
model.restore()
print("Start Fitting")
CASP11 = pnet.utils.load_CASP(11)
CASP11 = CASPALL.select_by_ID(CASP11._IDs)
CASP12 = pnet.utils.load_CASP(12)
CASP12 = CASPALL.select_by_ID(CASP12._IDs)
metrics = [pnet.utils.metrics.CoordinatesMetric()]
for i in range(20):
  model.fit(train, nb_epoch=1, checkpoint_interval=11498)

  print(model.evaluate(CASP11, metrics))
  print(model.evaluate(CASP12, metrics))