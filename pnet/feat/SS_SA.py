#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:33:55 2017

@author: zqwu
"""

import os
import pandas as pd
import numpy as np
import pickle
from pnet.utils import raptorx_ss, psipred_ss, psspred_ss, raptorx_sa, netsurfp_rsa

def generate_ss(dataset, mode="raptorx", reload=False):
  """Generate secondary structure prediction for a single sample(sequence)"""
  assert len(dataset.sequences) == 1, "Only support one sample"
  # Support psiblast, blastp
  data_dir = os.environ['PNET_DATA_DIR']
  ss_file = os.path.join(data_dir, 'SS_All/'+dataset.IDs[0]+'.ss')
  if os.path.exists(ss_file) and not reload:
    return load_ss(ss_file)
  if mode == "raptorx":
    ss_score = raptorx_ss(dataset)
  elif mode == "psspred":
    ss_score = psspred_ss(dataset)
  elif mode == "psipred":
    ss_score = psipred_ss(dataset)
  n_columns = ss_score.shape[0]
  print('Number of features for secondary structure: %i' % n_columns)
  return ss_score

def generate_sa(dataset, mode="raptorx", reload=False):
  """Generate solvent accessibility prediction for a single sample(sequence)"""
  assert len(dataset.sequences) == 1, "Only support one sample"
  # Support psiblast, blastp
  data_dir = os.environ['PNET_DATA_DIR']
  sa_file = os.path.join(data_dir, 'SA_All/'+dataset.IDs[0]+'.sa')
  if os.path.exists(sa_file) and not reload:
    return load_sa(sa_file)
  if mode == "raptorx":
    sa_score = raptorx_sa(dataset)
  elif mode == "netsurfp":
    sa_score = netsurfp_rsa(dataset)
  n_columns = sa_score.shape[0]
  print('Number of features for solvent accessibility: %i' % n_columns)
  return sa_score

def load_ss(path):
  with open(path, 'r') as f:
    ss = pickle.load(f)
  return ss

def load_sa(path):
  with open(path, 'r') as f:
    sa = pickle.load(f)
  return sa

def write_ss(ss_score, path):
  with open(path, 'w') as f:
    pickle.dump(ss_score, f)
  return

def write_sa(sa_score, path):
  with open(path, 'w') as f:
    pickle.dump(sa_score, f)
  return
