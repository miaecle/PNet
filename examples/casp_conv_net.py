#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:43:49 2017

@author: zqwu
"""

import pnet

CASP_all = pnet.utils.load_CASP_all()
n_samples = CASP_all.n_samples

train = CASP_all.select_by_index([i for i in range(int(0.8*n_samples))])
valid = CASP_all.select_by_index([i for i in range(int(0.8*n_samples), int(0.9*n_samples))])
test = CASP_all.select_by_index([i for i in range(int(0.9*n_samples), n_samples)])

for dataset in [valid]:
  dataset.build_features(['raw', 'MSA', 'SS', 'SA'])
  dataset.build_labels()

batch_size = 2
n_features = valid.n_features

model = pnet.models.ConvNetContactMap(
    n_res_feat=n_features,
    filter_size_1D=[51],
    n_filter_1D=[50],
    filter_size_2D=[25],
    n_filter_2D=[50],
    learning_rate=1e-3,
    batch_size=batch_size,
    use_queue=False,
    mode='classification')

model.fit(valid)