#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:40:00 2017

@author: zqwu
"""

import os
import pandas as pd
import numpy as np
from pnet.data import SequenceDataset
from pnet.data import merge_datasets

def load_sequence(path):
  """ Load sequence csv file """
  df = pd.read_csv(path)
  IDs = df['ID'].tolist()
  pdb_paths = df['pdb'].tolist()
  Seqs = df['sequence'].tolist()
  lengths = set([len(IDs), len(pdb_paths), len(Seqs)])
  assert len(lengths) == 1
  return SequenceDataset(IDs, sequences=Seqs, pdb_paths=pdb_paths, raw=None)

def load_raw_sequence(path):
  """ Load raw fasta file """
  with open(path, 'r') as f:
    lines = f.readlines()
  IDs = []
  raws = []
  ID = None
  raw = []
  for line in lines:
    if line[0] == '>':
      if ID != None:
        IDs.append(ID)
        ID = None
      if len(raw) > 0:
        raws.append(raw)
        raw = []
      ID = line[1:6]
      raw.append(line)
    elif line[0] == '\n':
      pass
    else:
      raw.append(line)
  if ID != None:
    IDs.append(ID)
  if len(raw) > 0:
    raws.append(raw)
  return SequenceDataset(IDs, sequences=None, pdb_paths=None, raw=raws)

def load_CASP(number, raw=False):
  """ Load CASP set """
  datasets_path = os.environ['PNET_DATA_DIR']
  assert int(number) in range(5,13), 'CASP' + str(int(number)) + ' is not supported'
  path = os.path.join(datasets_path, 'CASP'+str(int(number)))
  if raw:
    path = os.path.join(path, 'casp' + str(int(number)) + '.seq')
    return load_raw_sequence(path)
  else:
    path = os.path.join(path, 'casp' + str(int(number)) + '_seq.csv')
    return load_sequence(path)

def load_CASP_all(raw=False):
  """ Load all prepared CASP samples (5~12) """
  CASP_series = [5, 6, 7, 8, 9, 10, 11, 12]
  datasets = [load_CASP(i, raw=raw) for i in CASP_series]
  return merge_datasets(datasets)

def load_sample(ID):
  """ Load sample with specific ID """
  if not ID is list:
    ID = [ID]
  CASP_all = load_CASP_all(raw=False)
  return CASP_all.select_by_ID(ID)

def write_dataset(dataset, path):
  """ Write SequenceDataset to file(fasta)"""
  dataset.build_raw()
  with open(path, 'w') as f:
    num_samples = dataset.get_num_samples()
    for i in range(num_samples):
      f.writelines(dataset.raw[i])
      f.writelines(['\n', '\n'])
  return None

def write_sequence(sequences, path):
  """ Generate and write temporary SequenceDataset to file(fasta)"""
  if not sequences is list:
    sequences = [sequences]
  with open(path, 'w') as f:
    num_samples = len(sequences)
    for i in range(num_samples):
      f.writelines(['>'+'TEMP'+str(i%10)+'\n', sequences[i] + '\n'])
      f.writelines(['\n', '\n'])