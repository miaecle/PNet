#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:01:43 2018

@author: zqwu
"""

import pnet
import os

# Take CAMEO as an example
cameo = pnet.utils.load_CAMEO()

for i in reversed(range(0, cameo.n_samples)):
  print("On %d: %s" % (i, cameo._IDs[i]))
  dataset = cameo.select_by_index([i])
  pnet.feat.SS_SA.generate_sa(dataset)
  pnet.feat.SS_SA.generate_ss(dataset)
  _, path = pnet.utils.homology_search.hhblits_local(dataset, evalue=0.001, num_iterations=3)
  os.system('cp '+ path + ' /home/zqwu/PNet/datasets/MSA_ALL/'+dataset._IDs[0] + '.fas')
  pnet.feat.twoD_features.CCMpred(dataset)
  pnet.feat.twoD_features.MI_MCP(dataset)  
