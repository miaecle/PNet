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
train.build_2D_features(feat_list=['CCMpred', 'MI_MCP'], path=data_dir_train)
train.build_labels(path=data_dir_train, weight_base=50., weight_adjust=0.1, binary=True)

CASPALL = pnet.utils.load_CASP_all()
data_dir_valid = os.path.join(os.environ['PNET_DATA_DIR'], 'CASPALL')
CASPALL.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_valid)
CASPALL.build_2D_features(feat_list=['CCMpred', 'MI_MCP'], path=data_dir_valid)
CASPALL.build_labels(path=data_dir_valid, weight_base=50., weight_adjust=0.1, binary=True)

CAMEO = pnet.utils.load_CAMEO()
data_dir_cameo = os.path.join(os.environ['PNET_DATA_DIR'], 'CAMEO')
CAMEO.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_cameo)
CAMEO.build_2D_features(feat_list=['CCMpred', 'MI_MCP'], path=data_dir_cameo)
CAMEO.build_labels(path=data_dir_cameo, weight_base=50., weight_adjust=0.1, binary=True)

CASP11 = pnet.utils.load_CASP(11)
CASP11 = CASPALL.select_by_ID(CASP11._IDs)
CASP12 = pnet.utils.load_CASP(12)
CASP12 = CASPALL.select_by_ID(CASP12._IDs)

batch_size = 1
n_features = CASPALL.n_features
metrics = [pnet.utils.Metric(pnet.utils.top_k_accuracy(5), mode='classification')]
metrics2 = [pnet.utils.Metric(pnet.utils.top_k_accuracy(10), mode='classification')]

model_dir = os.path.join(os.environ['PNET_DATA_DIR'], '../saved_models/EncDec/')
model = pnet.models.EncodeDecodeContactMap(
    n_pool_layers=4,
    init_n_filter=32,
    n_res_feat=n_features,
    learning_rate=1e-3,
    learning_rate_decay=0.95,
    batch_size=batch_size,
    use_queue=False,
    uppertri=True,
    mode='classification',
    n_batches=None,
    oneD_loss=None,
    model_dir=model_dir)

model.build()
model.restore(os.path.join(model_dir, 'model-1'))
#model.fit(train, nb_epoch=25, checkpoint_interval=11498)
print(model.evaluate(CASP11, metrics))
print(model.evaluate(CASP12, metrics))
print(model.evaluate(CAMEO, metrics))
print(model.evaluate(CASP11, metrics2))
print(model.evaluate(CASP12, metrics2))
print(model.evaluate(CAMEO, metrics2))
