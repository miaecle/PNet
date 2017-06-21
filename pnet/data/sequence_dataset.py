#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:58:18 2017

@author: zqwu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import os
import pandas as pd
import mdtraj as md
import Bio
import pnet
from Bio.PDB import PDBList

def merge_datasets(datasets):
  dataset = datasets[0]
  for i in range(1, len(datasets)):
    dataset.extend(datasets[i])
  return dataset

class SequenceDataset(object):
  """
  Dataset class for protein sequences
  """
  def __init__(self, IDs, sequences=None, pdb_paths=None, raw=None, load_pdb=False, keep_all=False):
    """Hold information of IDs, sequences, path of pdb files and raw inputs"""
    self._IDs = list(IDs)
    self.n_samples = len(IDs)
    self.keep_all = keep_all
    self.load_pdb = load_pdb
    if sequences is None:
      self._sequences = [None] * self.n_samples
    else:
      self._sequences = list(sequences)
    if pdb_paths is None:
      self._pdb_paths = [None] * self.n_samples
    else:
      self._pdb_paths = list(pdb_paths)
    if raw is None:
      self._raw = [None] * self.n_samples
    else:
      self._raw = list(raw)
    if self.load_pdb:
      self.load_structures()
    self.X_built = False
    self.y_built = False

  def load_structures(self):
    data_dir = os.environ['PNET_DATA_DIR']
    save_dir = os.path.join(data_dir, 'PDB')
    ts = []
    resseqs = []
    pdbl = PDBList()
    for i, path in enumerate(self._pdb_paths):
      if self._IDs[i][0] == "T":
        if path == 'nan' or pd.isnull(path) or path is None:
          ts.append(None)
          resseqs.append(None)
        else:
          path = os.path.join(data_dir, path)
          t = md.load_pdb(path)
          resseq_pos = 1
          if not self.keep_all:
            atoms_to_keep = [a for a in t.topology.atoms if a.name == 'CB']
            resseqs.append(np.array([a.residue.resSeq for a in atoms_to_keep])-resseq_pos)
            index = [a.index for a in atoms_to_keep]
            t.restrict_atoms(index)
          else:
            resseqs.append(np.array([a.residue.resSeq for a in t.topology.atoms])-resseq_pos)
          ts.append(t)
      else:
        PDB_ID = self._IDs[i][:4]
        chain_ID = ord(self._IDs[i][4]) - ord('A')
        # fetch PDB from website
        pdbfile = pdbl.retrieve_pdb_file(PDB_ID, pdir=save_dir)
        t = md.load_pdb(pdbfile)
        chain = t.topology.chain(chain_ID)
        # Calibrate the starting position of chain
        code_start = [chain.residue(j).code for j in range(10)]
        resseq_start = [chain.residue(j).resSeq for j in range(10)]
        resseq_pos = self.calibrate_resseq(self._sequences[i], code_start, resseq_start)
        if not self.keep_all:
          atoms_to_keep = [a for a in chain.atoms if a.name == 'CB']
          resseqs.append(np.array([a.residue.resSeq for a in atoms_to_keep])-resseq_pos)
          index = [a.index for a in atoms_to_keep]
          t.restrict_atoms(index)
        else:
          resseqs.append(np.array([a.residue.resSeq for a in chain.atoms])-resseq_pos)
          index = [a.index for a in chain.atoms]
          t.restrict_atoms(index)
        ts.append(t)
    self._structures = ts
    self._resseqs = resseqs
    self.extract_coordinates()
    self.load_pdb = True

  @staticmethod
  def calibrate_resseq(seq, code_start, resseq_start):
    for i, res in enumerate(list(seq)):
      if res == code_start[0]:
        resseq_pos = resseq_start[0] - i
        if list(np.array(list(seq))[np.array(resseq_start[1:])-resseq_pos]) \
            == code_start[1:]:
          return resseq_pos


  def extract_coordinates(self):
    self.xyz = []
    for i, structure in enumerate(self._structures):
      if structure is None:
        self.xyz.append(None)
      else:
        coordinate = np.zeros((len(self._sequences[i]),3))
        coordinate[self._resseqs[i], :] = structure.xyz
        self.xyz.append(coordinate)

  def get_num_samples(self):
    return self.n_samples

  def extend(self, extension_dataset):
    """Add in another dataset, duplicate not included"""
    add_on = []
    for i, ID in enumerate(extension_dataset.IDs):
      if ID in self.IDs:
        pass
      else:
        add_on.append(i)
    add_on = np.array(add_on)
    IDs_add = list(np.array(extension_dataset.IDs)[add_on])
    pdb_paths_add = list(np.array(extension_dataset.pdb_paths)[add_on])
    sequences_add = list(np.array(extension_dataset.sequences)[add_on])
    raw_add = list(np.array(extension_dataset.raw)[add_on])
    self._IDs.extend(IDs_add)
    self._pdb_paths.extend(pdb_paths_add)
    self._sequences.extend(sequences_add)
    self._raw.extend(raw_add)
    if self.load_pdb:
      if not extension_dataset.load_pdb:
        extension_dataset.load_structures()
      structures_add = list(np.array(extension_dataset.structures)[add_on])
      resseqs_add = list(np.array(extension_dataset.resseqs)[add_on])
      xyz_add = list(np.array(extension_dataset.xyz)[add_on])
      self._structures.extend(structures_add)
      self._resseqs.extend(resseqs_add)
      self.xyz.extend(xyz_add)
    self.n_samples = self.n_samples + len(add_on)


  def build_from_raw(self, IDs=None):
    """Extract sequences from raw input(.seq, .fas)"""
    if IDs == None:
      index = range(self.get_num_samples())
    else:
      index = []
      for i, ID in enumerate(self._IDs):
        if ID in IDs:
          index.append(i)
    for i in index:
      if self._sequences[i] != None:
        pass
      elif self._raw[i] == None:
        print('No raw data available for ' + self._IDs[i])
      else:
        assert self._raw[i][0][0] == '>'
        sequence = ''
        for j in range(1, len(self._raw[i])):
          sequence = sequence + self._raw[i][j][:-1]
        self._sequences[i] = sequence

  def build_raw(self, IDs=None):
    """Build raw output(.seq, .fas) from sequence"""
    if IDs == None:
      index = range(self.get_num_samples())
    else:
      index = []
      for i, ID in enumerate(self._IDs):
        if ID in IDs:
          index.append(i)
    for i in index:
      if self._raw[i] != None:
        pass
      elif self._sequences[i] == None:
        print('No sequence available for ' + self._IDs[i])
      else:
        self._raw[i] = []
        self._raw[i].append('>' + self._IDs[i] + '\n')
        self._raw[i].append(self._sequences[i] + '\n')

  def build_residue_contact(self, binary=False, threshold=0.8):
    """ Build residue-residue contact map based on 3D coordinates"""
    if not self.load_pdb:
      self.load_structures()
    self.RRs = []
    for i, coordinate in enumerate(self.xyz):
      if coordinate is None:
        self.RRs.append(None)
      else:
        num_residues = coordinate.shape[0]
        RR = np.zeros((num_residues, num_residues, 3))
        RR[:,:,:] = coordinate
        RR = np.sqrt(np.sum(np.square(RR - np.transpose(RR, (1, 0, 2))), axis=2))
        invalid_index = np.delete(np.arange(num_residues), self._resseqs[i])
        RR[invalid_index, :] = 0
        RR[:, invalid_index] = 0
        if binary:
          RR = np.asarray(-RR + threshold > 0, dtype=bool)
          RR[invalid_index, :] = False
          RR[:, invalid_index] = False
        self.RRs.append(RR)

  @property
  def IDs(self):
    """ Get IDs as np.array"""
    return np.array(self._IDs)

  @property
  def sequences(self):
    """ Get sequences as np.array"""
    return np.array(self._sequences)

  @property
  def pdb_paths(self):
    """ Get pdb paths as np.array"""
    return np.array(self._pdb_paths)

  @property
  def raw(self):
    """ Get raw fasta as np.array"""
    return np.array(self._raw)

  @property
  def structures(self):
    """ Get structures as np.array (of mdtraj.Trajectory object) """
    if not self.load_pdb:
      self.load_structures()
    return np.array(self._structures)

  @property
  def resseqs(self):
    """ Get residue index of all CB atoms """
    if not self.load_pdb:
      self.load_structures()
    return np.array(self._resseqs)

  def select_by_index(self, index):
    """Creates a new dataset from a selection of index.
    """
    IDs = self.IDs[index]
    sequences = self.sequences[index]
    pdb_paths = self.pdb_paths[index]
    raw = self.raw[index]
    result = SequenceDataset(IDs, sequences, pdb_paths, raw, load_pdb=False)
    if self.load_pdb:
      result._structures = [self._structures[i] for i in index]
      result._resseqs = [self._resseqs[i] for i in index]
      result.xyz = [self.xyz[i] for i in index]
      result.load_pdb = True
    return result

  def select_by_ID(self, IDs_selection):
    """Creates a new dataset from a selection of IDs.
    """
    index = []
    for i, ID in enumerate(self._IDs):
      if ID in IDs_selection:
        index.append(i)
    return self.select_by_index(index)

  def fetch_features(self, feat='MSA'):
    if feat == 'MSA':
      feat = pnet.feat.generate_msa
    elif feat == 'SS':
      feat = pnet.feat.generate_ss
    elif feat == 'SA':
      feat = pnet.feat.generate_sa
    elif feat == 'raw':
      feat = pnet.feat.generate_raw
    return [feat(self.select_by_index([i])) for i in range(self.n_samples)]

  def build_features(self, feat_list):
    if not feat_list.__class__ is list:
      feat_list = [feat_list]
    n_feats = len(feat_list)
    X = [self.fetch_features(feat=feature) for feature in feat_list]
    self.X = [np.concatenate([X[i][j] for i in range(n_feats)], axis=1) for j in range(self.n_samples)]
    self.X_built = True

  def build_labels(self, task='RR'):
    if task == 'RR':
      self.build_residue_contact()
      self.y = self.RRs
      self.w = [np.ones_like(RR) for RR in self.RRs]
      for i, weight_matrix in enumerate(self.w):
        weight_matrix[:, self._resseqs[i]] = 0
        weight_matrix[self._resseqs[i], :] = 0
    else:
      self.y = self.xyz
      self.w = [1 for i in range(self.n_samples)]
    self.y_built = True
    
  def iterbatches(self,
                  batch_size=None,
                  deterministic=False,
                  pad_batches=False):
    """Object that iterates over minibatches from the dataset.
    """
    assert self.X_built, "Dataset not ready for training, features must be built"
    assert self.y_built, "Dataset not ready for training, labels must be built"
    def iterate(dataset, batch_size, deterministic, pad_batches):
      n_samples = dataset.get_num_samples()
      if not deterministic:
        sample_perm = np.random.permutation(n_samples)
      else:
        sample_perm = np.arange(n_samples)
      if batch_size is None:
        batch_size = n_samples
      interval_points = np.linspace(
          0, n_samples, np.ceil(float(n_samples) / batch_size) + 1, dtype=int)
      for j in range(len(interval_points) - 1):
        indices = range(interval_points[j], interval_points[j + 1])
        perm_indices = sample_perm[indices]
        out_X = [self.X[i] for i in perm_indices]
        out_y = [self.y[i] for i in perm_indices]
        out_w = [self.w[i] for i in perm_indices]
        if pad_batches:
          out_X, out_y, out_w = self.pad_batch(batch_size, out_X, out_y, out_w)
        yield out_X, out_y, out_w
    return iterate(self, batch_size, deterministic, pad_batches)

  @staticmethod
  def pad_batch(batch_size, X, y, w):
    """Pads batch to have size of batch_size
    """
    num_samples = len(X)
    assert num_samples <= batch_size
    if num_samples < batch_size:
      X.extend([None] * (batch_size - num_samples))
      y.extend([None] * (batch_size - num_samples))
      w.extend([None] * (batch_size - num_samples))
    return X, y, w
  
  def itersamples(self):
    """Object that iterates over the samples in the dataset.
    """
    def sample_iterate(dataset):
      n_samples = dataset.get_num_samples()
      for i in range(n_samples):
        yield self.X[i], self.y[i], self.w[i]
    return sample_iterate(self)
