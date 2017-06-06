#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:04:52 2017

@author: zqwu
"""
import os
import mdtraj as md

def load_pdb(dataset, keep_all=False):
  data_dir = os.environ['PNET_DATA_DIR']
  ts = []
  for path in dataset.pdb_paths:
    if path == 'nan':
      ts.append(None)
    else:
      path = os.path.join(data_dir, path)
      t = md.load_pdb(path)
      if not keep_all:
        atoms_to_keep = [a.index for a in t.topology.atoms if a.name == 'CA']
        t.restrict_atoms(atoms_to_keep)
      ts.append(t)
