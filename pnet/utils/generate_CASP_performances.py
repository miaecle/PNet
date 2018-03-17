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

dataset = pnet.utils.load_CASP(11)

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
if not os.path.exists('/home/zqwu/PNet/temp'):
  os.mkdir('/home/zqwu/PNet/temp')
os.chdir('/home/zqwu/PNet/temp')
for i in range(len(dataset._IDs)):
  ID = dataset._IDs[i]
  label_file = dataset._pdb_paths[i]
  
  if not label_file == label_file:
    top_performances[ID] = np.zeros((3, 0), dtype=object)
    continue
  os.system('wget http://www.predictioncenter.org/download_area/CASP11/SUMMARY_TABLES/'+ID+'.txt')
  os.system('wget http://www.predictioncenter.org/download_area/CASP11/predictions/'+ID+'.tar.gz')
  if not os.path.exists(ID+'.txt'):
    top_performances[ID] = np.zeros((3, 0), dtype=object)
    continue
  
  tar = tarfile.open(ID+'.tar.gz', "r:gz")
  tar.extractall()
  tar.close()
  
  top_performances[ID] = []
  
  label_file = os.path.join(os.environ['PNET_DATA_DIR'], label_file)
  #Load reference structure
  n_residues = len(dataset._sequences[i])
  t = md.load_pdb(label_file)
  #assert t.n_chains == 1
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
  ref_resseq = np.array([a.residue.resSeq for a in atoms_to_keep]) - resseq_pos
  index = [a.index for a in atoms_to_keep]
  t.restrict_atoms(index)
  # Calibrate the starting position of chain
  ref_coordinate = np.zeros((n_residues,3))
  ref_w = np.zeros((n_residues,))
  ref_coordinate[ref_resseq, :] = t.xyz
  ref_w[ref_resseq] = 1
  
  
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
    if len(set(ref_resseq) - set(resseq)) > 0:
      print("missing residues in "+model)
      top_performances[ID].append((model, None, flag))
      continue
    index = [a.index for a in atoms_to_keep]
    t.restrict_atoms(index)
    # Calibrate the starting position of chain
    coordinate = np.zeros((n_residues,3))
    coordinate[resseq, :] = t.xyz
    
    score = metric.compute_metric([ref_coordinate], [coordinate], [ref_w])[0]
    top_performances[ID].append((model, float(score), flag))
  top_performances[ID] = np.stack(top_performances[ID], 0)

os.chdir('/home/zqwu/PNet/examples')
with open('./CASP11_top.pkl', 'w') as f:
  pickle.dump(top_performances, f)