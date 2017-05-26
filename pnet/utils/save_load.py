# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
from pnet.dataset import SequenceDataset
from pnet.dataset import merge_datasets

def load_sequence(path):
  df = pd.read_csv(path)
  IDs = df['ID'].tolist()
  pdb_paths = df['pdb'].tolist()
  Seqs = df['sequence'].tolist()
  lengths = set([len(IDs), len(pdb_paths), len(Seqs)])
  assert len(lengths) == 1
  return SequenceDataset(IDs, sequences=Seqs, pdb_paths=pdb_paths, raw=None)

def load_raw_sequence(path):
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
  CASP_series = [5, 6, 7, 8, 9, 10, 11, 12]
  datasets = [load_CASP(i, raw=raw) for i in CASP_series]
  return merge_datasets(datasets)

def load_sample(ID):
  if not ID is list:
    ID = [ID]
  CASP_all = load_CASP_all(raw=False)
  return CASP_all.select_by_ID(ID)

def write_sample(dataset, path):
  dataset.build_raw()
  