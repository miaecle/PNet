#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 18:27:16 2018

@author: zqwu
"""

import pnet
import os
import pickle
import tarfile
import mdtraj as md
import numpy as np
import pandas as pd

CASPALL = pnet.utils.load_CASP_all()
data_dir_valid = os.path.join(os.environ['PNET_DATA_DIR'], 'CASPALL')
CASPALL.build_features(['raw', 'MSA', 'SS', 'SA'], path=data_dir_valid)
CASPALL.build_labels(path=data_dir_valid, weight_base=50., weight_adjust=0.1, binary=True)

dataset = pnet.utils.load_CASP(12)
dataset = CASPALL.select_by_ID(dataset._IDs)

w_all = []
y_all = []

for (_, _, _, _, oneD_y_b, oneD_w_b) in dataset.iterbatches(
    use_contact_prob=False,
    batch_size=1,
    deterministic=True,
    pad_batches=False):
    y_all.extend(oneD_y_b)
    w_all.extend(oneD_w_b)

def _calibrate_resseq(seq, code_start, resseq_start):
  """find the difference in index between sequence and pdb file"""
  for i, res in enumerate(list(seq)):
    if res == code_start[0]:
      resseq_pos = resseq_start[0] - i
      if sum(np.array(list(seq))[np.array(resseq_start[1:])-resseq_pos] == \
             np.array(code_start[1:])) >= (len(code_start) - 1) * 0.98:
        return resseq_pos
  return False

top_performances = {}
metric = pnet.utils.metrics.CoordinatesMetric()
os.chdir('/home/zqwu/PNet/temp')
for i in range(len(dataset._IDs)):
  ID = dataset._IDs[i]
  if list(np.unique(w_all[i])) == [0.0]:
    top_performances[ID] = np.zeros((3, 0), dtype=object)
    continue
  os.system('wget http://www.predictioncenter.org/download_area/CASP12/SUMMARY_TABLES/'+ID+'.txt')
  os.system('wget http://www.predictioncenter.org/download_area/CASP12/predictions/'+ID+'.tar.gz')
  if not os.path.exists(ID+'.txt'):
    top_performances[ID] = np.zeros((3, 0), dtype=object)
    continue
  
  tar = tarfile.open(ID+'.tar.gz', "r:gz")
  tar.extractall()
  tar.close()
  
  top_performances[ID] = []
  df = pd.read_table(ID+'.txt', delim_whitespace=True)
  for order, model in enumerate(df['Model']):
    if df['GR#'][order][-1] == 's':
      flag = 1
    else:
      flag = 0
    
    filename = './'+ID+'/'+model
    if not os.path.exists(filename):
      print("File not existing "+model)
      top_performances[ID].append((model, None, flag))
      continue
      
    n_residues = len(dataset._sequences[i])
    t = md.load_pdb(filename)
    assert t.n_chains == 1
    chain = t.topology.chain(0)
    resNames = np.unique(list(dataset._sequences[i]))
    code_start = [chain.residue(j).code for j in range(chain.n_residues) if chain.residue(j).code in resNames]
    resseq_start = [chain.residue(j).resSeq for j in range(chain.n_residues) if chain.residue(j).code in resNames]
    resseq_pos = _calibrate_resseq(dataset._sequences[i], code_start, resseq_start)
    # resseq_pos should always be non-negative
    assert isinstance(resseq_pos, int), "Conflict in sequences"
    atoms_to_keep = []
    for res in chain.residues:
      try:
        if res.code == 'G':
          atoms_to_keep.append(res.atom('CA'))
        else:
          atoms_to_keep.append(res.atom('CB'))
      except KeyError:
        print("%s residue number: %d %s" % (dataset._IDs[i], res.resSeq-resseq_pos, res.code))
    resseq = np.array([a.residue.resSeq for a in atoms_to_keep]) - resseq_pos
    missing_resseqs = np.delete(np.arange(n_residues), resseq)
    if np.any(w_all[i][missing_resseqs] == 1) == True:
      print("missing residues in "+model)
      top_performances[ID].append((model, None, flag))
      continue
    index = [a.index for a in atoms_to_keep]
    t.restrict_atoms(index)
    # Calibrate the starting position of chain
    coordinate = np.zeros((n_residues,3))
    coordinate[resseq, :] = t.xyz
    
    score = metric.compute_metric([y_all[i]], [coordinate], [w_all[i]])[0]
    top_performances[ID].append((model, float(score), flag))
  top_performances[ID] = np.stack(top_performances[ID], 0)

os.chdir('/home/zqwu/PNet/examples')
with open('./CASP12_top.pkl', 'w') as f:
  pickle.dump(top_performances, f)