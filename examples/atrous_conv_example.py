#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:43:49 2017

@author: zqwu

Example script using saved atrous-conv model to predict targets in CASP11, CASP12, CAMEO.

To run this code, please download the feature files(or generating features by HHblits, CCMpred and RaptorX):
  CASP: https://s3-us-west-1.amazonaws.com/deepchem.io/featurized_datasets/CASPALL.tar.gz
  CAMEO: https://s3-us-west-1.amazonaws.com/deepchem.io/featurized_datasets/CAMEO.tar.gz
  PDB50cut: https://s3-us-west-1.amazonaws.com/deepchem.io/featurized_datasets/PDB50cut.tar.gz
Please decompress into the datasets folder, as defined by the environmental variable: PNET_DATA_DIR

"""
import deepchem as dc
import numpy as np
import pnet
import os

train = pnet.utils.load_PDB50_cut()
data_dir_train = os.path.join(os.environ['PNET_DATA_DIR'], 'PDB50cut')
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

model_dir = os.path.join(os.environ['PNET_DATA_DIR'], '../saved_models/AtrousConv/')
model = pnet.models.AtrousConvContactMap(
    n_res_feat=n_features,
    learning_rate=1e-3,
    learning_rate_decay=0.99,
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
