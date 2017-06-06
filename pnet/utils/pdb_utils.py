#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:04:52 2017

@author: zqwu
"""
import os
import mdtraj as md
from pnet.data import StructureDataset

def load_pdb(dataset, keep_all=False):
  data_dir = os.environ['PNET_DATA_DIR']
  ts = []
  resseqs = []
  for path in dataset.pdb_paths:
    if path == 'nan':
      ts.append(None)
    else:
      path = os.path.join(data_dir, path)
      t = md.load_pdb(path)
      if not keep_all:
        atoms_to_keep = [a for a in t.topology.atoms if a.name == 'CB']
        resseqs.append([a.residue.resSeq for a in atoms_to_keep])
        index = [a.index for a in atoms_to_keep]
        t.restrict_atoms(index)
      else:
        resseqs.append([a.residue.resSeq for a in t.topology.atoms])
      ts.append(t)
  return StructureDataset.from_SequenceDataset(ts, resseqs, dataset)