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
import tempfile
from Bio.PDB import PDBList

def merge_datasets(datasets):
  """ Merge a list of datasets into one dataset """
  dataset = datasets[0]
  for i in range(1, len(datasets)):
    dataset.extend(datasets[i])
  return dataset

class SequenceDataset(object):
  """
  Dataset class for protein sequences and structures
  """
  def __init__(self, IDs, sequences=None, pdb_paths=None, raw=None, load_pdb=False, keep_all=False):
    """Hold information of IDs, sequences, path of pdb files and raw inputs

    Parameters
    ----------
    IDs: list
      list of strings including protein IDs in the dataset
    sequences: list, optional
      list of strings including protein sequences
    pdb_paths: list, optional
      list of strings including pdb paths for proteins in the dataset,
      protein with no structure available will have None as placeholder
    raw: list, optional
      list of strings including .seq file input for protein sequences
    load_pdb: bool, optional
      whether to load pdb structures and build residue-residue contact maps
    keep_all: bool, optional
      whether to keep coordinates of all atoms(False = only keep beta-carbon)
    """
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
    self.twoD_X_built = False
    self.y_built = False
    self.oneD_y_built = False
    self.X_on_disk = False
    self.twoD_X_on_disk = False
    self.y_on_disk = False
    self.oneD_y_on_disk = False
    self.contact_prob_built = False
    self.contact_prob_on_disk = False

  def load_structures(self, binary=False, threshold=0.8, weight_base=30., weight_adjust=0.1):
    """
    load pdb structures for samples
    generate 3D coordinates and residue-residue contact map
    """
    data_dir = os.environ['PNET_DATA_DIR']
    save_dir = os.path.join(data_dir, 'PDB')
    ts = []
    resseqs = []
    xyz = []
    xyz_weights = []
    phis = []
    psis = []
    phi_psi_weights = []
    RRs = []
    RR_weights = []
    pdbl = PDBList()

    for i, path in enumerate(self._pdb_paths):
      n_residues = len(self._sequences[i])
      if path == 'nan' or pd.isnull(path) or path is None:
        ts.append(None)
        resseqs.append(None)
        xyz.append(np.zeros((n_residues, 3)))
        xyz_weights.append(np.zeros((n_residues,)))
        phis.append(np.zeros((n_residues,)))
        psis.append(np.zeros((n_residues,)))
        phi_psi_weights.append(np.zeros((n_residues,)))
        RR = np.zeros((n_residues, n_residues))
        if binary:
          RRs.append(np.array(RR, dtype=bool))
        else:
          RRs.append(RR)
        RR_weight = np.zeros((n_residues, n_residues))
        RR_weights.append(RR_weight)
      else:
        if self._IDs[i][0] == "T":
          path = os.path.join(data_dir, path)
          # Load pdb
          t = md.load_pdb(path)
          # CASP samples all have only one chain
          chain = t.topology.chain(0)
        else:
          PDB_ID = self._IDs[i][:4]
          chain_ID = ord(self._IDs[i][4]) - ord('A')
          # fetch PDB from website
          pdbfile = pdbl.retrieve_pdb_file(PDB_ID, pdir=save_dir, file_format='pdb')
          t = md.load_pdb(pdbfile)
          chain = t.topology.chain(chain_ID)

        # Calibrate the starting position of chain
        resNames = np.unique(list(self._sequences[i]))
        code_start = [chain.residue(j).code for j in range(chain.n_residues) if chain.residue(j).code in resNames]
        resseq_start = [chain.residue(j).resSeq for j in range(chain.n_residues) if chain.residue(j).code in resNames]
        resseq_pos = self._calibrate_resseq(self._sequences[i], code_start, resseq_start)
        # resseq_pos should always be non-negative
        assert isinstance(resseq_pos, int), "Conflict in sequences"
        
        # Calculate phi, psi angles
        phi, psi, phi_psi_weight = self.calculate_phi_psi(chain, t, resseq_pos, n_residues)
        phis.append(phi)
        psis.append(psi)
        phi_psi_weights.append(phi_psi_weight)
        if not self.keep_all:
          # Only use beta-C to calculate residue-residue contact
          atoms_to_keep = []
          for res in chain.residues:
            try:
              if res.code == 'G':
                atoms_to_keep.append(res.atom('CA'))
              else:
                atoms_to_keep.append(res.atom('CB'))
            except KeyError:
              print("%s residue number: %d %s" % (self._IDs[i], res.resSeq-resseq_pos, res.code))
          # Residue position in the sequence
          resseq = np.array([a.residue.resSeq for a in atoms_to_keep]) - resseq_pos
          index = [a.index for a in atoms_to_keep]
          t.restrict_atoms(index)
        else:
          resseq = np.array([a.residue.resSeq for a in chain.atoms]) - resseq_pos
          index = [a.index for a in chain.atoms]
          t.restrict_atoms(index)

        # 3D-coordinates
        coordinate = np.zeros((n_residues,3))
        coordinate_weight = np.zeros((n_residues,))
        coordinate[resseq, :] = t.xyz
        coordinate_weight[resseq] = 1
        

        RR = np.zeros((n_residues, n_residues, 3))
        RR_weight = np.ones((n_residues, n_residues))

        RR[:,:,:] = coordinate
        # Pairwise distance calculation
        RR = np.sqrt(np.sum(np.square(RR - np.transpose(RR, (1, 0, 2))), axis=2))
        invalid_index = np.delete(np.arange(n_residues), resseq)
        RR[invalid_index, :] = 0
        RR[:, invalid_index] = 0
        RR_weight[invalid_index, :] = 0
        RR_weight[:, invalid_index] = 0
        if binary:
          # All contact within threshold will be set as True
          RR = np.asarray(-RR + threshold > 0, dtype=bool)
          RR[invalid_index, :] = False
          RR[:, invalid_index] = False
          RR_weight_adjust = np.abs(np.stack([np.arange(n_residues)] * n_residues, axis=0) -
                                    np.stack([np.arange(n_residues)] * n_residues, axis=1))
          RR_weight_adjust = RR.astype(float) * (weight_base + RR_weight_adjust * weight_adjust)
          RR_weight = RR_weight + RR_weight_adjust
          full_range = np.abs(np.stack([np.arange(n_residues)] * n_residues, axis=0) -
              np.stack([np.arange(n_residues)] * n_residues, axis=1))
          RR_weight = RR_weight * ((full_range - 6 ) >= 0).astype(float)
        else:
          full_range = np.abs(np.stack([np.arange(n_residues)] * n_residues, axis=0) -
              np.stack([np.arange(n_residues)] * n_residues, axis=1))
          RR_weight = RR_weight + np.clip(full_range * weight_adjust, 0, 0.5) + weight_base * np.exp(-1.25 * RR)
          RR_weight = RR_weight * ((full_range - 6 ) >= 0)
          RR_weight[invalid_index, :] = 0
          RR_weight[:, invalid_index] = 0
        ts.append(t)
        resseqs.append(resseq)
        xyz.append(coordinate)
        xyz_weights.append(coordinate_weight)
        RRs.append(RR)
        RR_weights.append(RR_weight)
    self._structures = ts
    self._resseqs = resseqs
    self.xyz = xyz
    self.xyz_weights = xyz_weights
    self.calculate_residue_distances(self.xyz)
    self.phis = phis
    self.psis = psis
    self.phi_psi_weights = phi_psi_weights
    self.RRs = RRs
    self.RR_weights = RR_weights
    self.load_pdb = True

  def calculate_residue_distances(self, xyz):
    self.distances = []
    self.dis_weights = []
    for i in range(self.n_samples):
      length = len(self._sequences[i])
      distance = np.zeros((length,))
      dis_weight = np.zeros((length,))
      if not xyz[i] is None:
        assert xyz[i].shape[0] == length
        for j in range(1, length-1):
          if np.all(xyz[i][j, :] == np.zeros((3,))) or \
             np.all(xyz[i][j+1, :] == np.zeros((3,))):
             pass
          else:
            distance[j] = np.linalg.norm(xyz[i][j+1, :] - xyz[i][j, :])
            dis_weight[j] = 1
      self.distances.append(np.array(distance))
      self.dis_weights.append(np.array(dis_weight))

  def calculate_phi_psi(self, chain, t, resseq_pos, n_res):
    indices1, angles1 = md.compute_phi(t)
    indices2, angles2 = md.compute_psi(t)
    restricted_residues = list(chain.residues)
    indices1 = [t.topology.atom(indice[0]).residue for indice in indices1]
    indices2 = [t.topology.atom(indice[0]).residue for indice in indices2]
    
    phi = [angles1[0, i] for i in range(len(angles1[0])) if indices1[i] in restricted_residues]
    phi_inds = np.array([indice.resSeq for indice in indices1 if indice in restricted_residues]) - resseq_pos
    psi = [angles2[0, i] for i in range(len(angles2[0])) if indices2[i] in restricted_residues]
    psi_inds = np.array([indice.resSeq for indice in indices2 if indice in restricted_residues]) - resseq_pos
    
    out = np.zeros((n_res, 2)) # phi, psi
    weight1 = np.zeros((n_res, 1))
    weight2 = np.zeros((n_res, 1))
    for i, ind in enumerate(phi_inds):
      out[ind, 0] = phi[i]
      weight1[ind] -= 1
    for i, ind in enumerate(psi_inds):
      out[ind, 1] = psi[i]
      weight2[ind] -= 1
    return out[:, 0], out[:, 1], (weight1*weight2)[:, 0]
    
    
  @staticmethod
  def _calibrate_resseq(seq, code_start, resseq_start):
    """find the difference in index between sequence and pdb file"""
    for i, res in enumerate(list(seq)):
      if res == code_start[0]:
        resseq_pos = resseq_start[0] - i
        if sum(np.array(list(seq))[np.array(resseq_start[1:])-resseq_pos] == \
               np.array(code_start[1:])) >= (len(code_start) - 1) * 0.98:
          return resseq_pos
    return False
  
  @staticmethod
  def _calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)
    
  def get_num_samples(self):
    """ return number of proteins in the dataset"""
    return self.n_samples

  def extend(self, extension_dataset):
    """Add in another dataset, duplicate not included"""
    add_on = []
    for i, ID in enumerate(extension_dataset.IDs):
      if ID in self.IDs:
        # Remove duplicate samples
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
      # Extend structure data
      if not extension_dataset.load_pdb:
        extension_dataset.load_structures()
      structures_add = list(np.array(extension_dataset.structures)[add_on])
      resseqs_add = list(np.array(extension_dataset.resseqs)[add_on])
      xyz_add = list(np.array(extension_dataset.xyz)[add_on])
      RRs_add = list(np.array(extension_dataset.RRs)[add_on])
      RR_weights_add = list(np.array(extension_dataset.RR_weights)[add_on])
      self._structures.extend(structures_add)
      self._resseqs.extend(resseqs_add)
      self.xyz.extend(xyz_add)
      self.RRs.append(RRs_add)
      self.RR_weights.append(RR_weights_add)
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
    """
    Build raw output(.seq, .fas) from sequence
    should be called before writing dataset
    """
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
    if isinstance(index, int):
      result = SequenceDataset([IDs], [sequences], [pdb_paths], [raw], load_pdb=False)
    else:
      result = SequenceDataset(IDs, sequences, pdb_paths, raw, load_pdb=False)
    if self.load_pdb:
      # Directly port structure data to selected dataset
      result._structures = [self._structures[i] for i in index]
      result._resseqs = [self._resseqs[i] for i in index]
      result.xyz = [self.xyz[i] for i in index]
      result.RRs = [self.RRs[i] for i in index]
      result.RR_weights = [self.RR_weights[i] for i in index]
      result.load_pdb = True
    if self.X_built:
      if self.X_on_disk:
        result.X = self.pick_Xyw(index, self.X, identifier='X')
        result.X_on_disk = True
      else:
        result.X = [self.X[i] for i in index]
      result.X_built = True
    if self.twoD_X_built:
      if self.twoD_X_on_disk:
        result.twoD_X = self.pick_Xyw(index, self.twoD_X, identifier='twoD_X')
        result.twoD_X_on_disk = True
      else:
        result.twoD_X = [self.twoD_X[i] for i in index]
      result.twoD_X_built = True
    if self.y_built:
      if self.y_on_disk:
        result.y = self.pick_Xyw(index, self.y, identifier='y')
        result.w = self.pick_Xyw(index, self.w, identifier='w')
        result.y_on_disk = True
      else:
        result.y = [self.y[i] for i in index]
        result.w = [self.w[i] for i in index]
      result.y_built = True
    if self.oneD_y_built:
      if self.oneD_y_on_disk:
        result.oneD_y = self.pick_Xyw(index, self.oneD_y, identifier='oneD_y')
        result.oneD_w = self.pick_Xyw(index, self.oneD_w, identifier='oneD_w')
        result.oneD_y_on_disk = True
      else:
        result.oneD_y = [self.oneD_y[i] for i in index]
        result.oneD_w = [self.oneD_w[i] for i in index]
      result.oneD_y_built = True
    if self.contact_prob_built:
      assert self.contact_prob_on_disk
      result.contact_prob = self.pick_Xyw(index, self.contact_prob, identifier='contact_prob')
      result.contact_prob_on_disk = True
      result.contact_prob_built = True
    return result

  def select_by_ID(self, IDs_selection):
    """ Creates a new dataset from a selection of IDs """
    index = []
    for i, ID in enumerate(self._IDs):
      if ID in IDs_selection:
        index.append(i)
    return self.select_by_index(index)

  def fetch_features(self, feat='MSA'):
    """ Fetch certain features for all samples as a list """
    if feat == 'MSA':
      feat = pnet.feat.generate_msa
    elif feat == 'SS':
      feat = pnet.feat.generate_ss
    elif feat == 'SA':
      feat = pnet.feat.generate_sa
    elif feat == 'raw':
      feat = pnet.feat.generate_raw
    return [feat(self.select_by_index([i])) for i in range(self.n_samples)]

  def fetch_2D_features(self, feat='CCMpred'):
    """ Fetch certain features for all samples as a list """
    if feat == 'CCMpred':
      feat = pnet.feat.CCMpred
    elif feat == 'MI_MCP':
      feat = pnet.feat.MI_MCP
    return feat(self)

  def build_features(self, feat_list, file_size=1000, reload=True, path=None):
    """ Build X based on specified list of features """
    if not path is None:
      path = os.path.join(path, '_'.join(feat_list))
      if not os.path.exists(path):
        os.mkdir(path)
      path = os.path.join(path, 'X')
      if reload and os.path.exists(path + '0.joblib'):
        self.X = self.load_joblib(path)
        self.X_built = True
        self.X_on_disk = True
        return
    if not feat_list.__class__ is list:
      feat_list = [feat_list]
    n_feats = len(feat_list)
    X = [self.fetch_features(feat=feature) for feature in feat_list]
    self.X = [np.concatenate([X[i][j] for i in range(n_feats)], axis=1) for j in range(self.n_samples)]
    self.X_built = True
    if reload and not path is None:
      self.save_joblib(self.X, path, file_size=file_size)

  def build_2D_features(self, feat_list=['CCMpred', 'MI_MCP'], file_size=100, reload=True, path=None):
    """ Build X based on specified list of features """
    if not path is None:
      path = os.path.join(path, '_'.join(feat_list))
      if not os.path.exists(path):
        os.mkdir(path)
      path = os.path.join(path, 'twoD_X')
      if reload and os.path.exists(path + '0.joblib'):
        self.twoD_X = self.load_joblib(path)
        self.twoD_X_built = True
        self.twoD_X_on_disk = True
        return
    if not feat_list.__class__ is list:
      feat_list = [feat_list]
    twoD_X = [self.fetch_2D_features(feat=feature) for feature in feat_list]
    self.twoD_X = [np.concatenate([twoD_X[i][j] for i in range(len(feat_list))], axis=2) for j in range(self.n_samples)]
    self.twoD_X_built = True
    if reload and not path is None:
      self.save_joblib(self.twoD_X, path, file_size=file_size)
      
  @property
  def n_features(self):
    """ Return number of features for each sample,
    can only be called after building X """
    assert self.X_built, "X not built"
    if self.X_on_disk:
      X_samp = pnet.utils.load_from_joblib(self.X[0])
      return X_samp[0].shape[1]
    else:
      return self.X[0].shape[1]

  def build_labels(self, task='RR', binary=True, threshold=0.8, weight_base=30., 
                   weight_adjust=0.1, file_size=100, reload=True, path=None):
    """ Build labels(y and w) for all samples """
    self.build_1D_labels(reload=reload, path=path)
    if not path is None:
      if binary:
        path = os.path.join(path, 'binary_'+str(threshold)+'_'+str(weight_base)+'_'+str(weight_adjust))
        if not os.path.exists(path):
          os.makedirs(path)
      else:
        path = os.path.join(path, 'continuous_'+str(weight_base)+'_'+str(weight_adjust))
        if not os.path.exists(path):
          os.makedirs(path)
      path_y = os.path.join(path, 'y')
      path_w = os.path.join(path, 'w')
      if reload and os.path.exists(path_y + '0.joblib') and os.path.exists(path_w + '0.joblib'):
        self.y = self.load_joblib(path_y)
        self.w = self.load_joblib(path_w)
        self.y_built = True
        self.y_on_disk = True
        return
    if not self.load_pdb:
      self.load_structures(binary=binary,
                           threshold=threshold,
                           weight_base=weight_base,
                           weight_adjust=weight_adjust)
    if task == 'RR':
      self.y = self.RRs
      self.w = self.RR_weights
    else:
      self.y = self.xyz
      self.w = [1 for i in range(self.n_samples)]
    self.y_built = True
    if reload and not path is None:
      self.save_joblib(self.y, path_y, file_size=file_size)
      self.save_joblib(self.w, path_w, file_size=file_size)
      self.y = self.load_joblib(path_y)
      self.w = self.load_joblib(path_w)
      self.y_on_disk = True
    
  def build_1D_labels(self, file_size=500, reload=True, path=None):
    """ Build 1D labels(y and w) for all samples:  distances, psis, phis
    
    """
    if not path is None:
      path = os.path.join(path, 'oneD_label')
      if not os.path.exists(path):
        os.makedirs(path)
      path_y = os.path.join(path, 'oneD_y')
      path_w = os.path.join(path, 'oneD_w')
      if reload and os.path.exists(path_y + '0.joblib') and os.path.exists(path_w + '0.joblib'):
        self.oneD_y = self.load_joblib(path_y)
        self.oneD_w = self.load_joblib(path_w)
        self.oneD_y_built = True
        self.oneD_y_on_disk = True
        return
    if not self.load_pdb:
      self.load_structures()
    #self.oneD_y = [np.stack([self.distances[i], self.phis[i], self.psis[i]], axis=1) for i in range(self.n_samples)]
    self.oneD_y = self.xyz
    self.oneD_w = self.xyz_weights
    for i, y in enumerate(self.oneD_y):
      if y is None:
        self.oneD_y[i] = np.zeros((len(self._sequences[i]), 3))
    self.oneD_y_built = True
    if reload and not path is None:
      self.save_joblib(self.oneD_y, path_y, file_size=file_size)
      self.save_joblib(self.oneD_w, path_w, file_size=file_size)
      self.oneD_y = self.load_joblib(path_y)
      self.oneD_w = self.load_joblib(path_w)
      self.oneD_y_on_disk = True

  def build_contact_prob(self, path=None):
    if not path is None:
      path = os.path.join(path, 'contact_prob/contact_prob')
      if os.path.exists(path+'0.joblib'):
        self.contact_prob = self.load_joblib(path)
        self.contact_prob_built = True
        self.contact_prob_on_disk = True
    
  @staticmethod
  def save_joblib(data, path, file_size=1000):
    total_length = len(data)
    file_num = total_length // file_size
    for i in range(file_num):
      path_save = path + str(i) + '.joblib'
      data_save = data[i*file_size:(i+1)*file_size]
      pnet.utils.save_to_joblib(data_save, path_save)
    if file_num * file_size < total_length:
      path_save = path + str(file_num) + '.joblib'
      data_save = data[file_num*file_size:]
      pnet.utils.save_to_joblib(data_save, path_save)

  @staticmethod
  def load_joblib(path):
    i = 0
    files = []
    while os.path.exists(path+str(i)+'.joblib'):
      files.append(path+str(i)+'.joblib')
      i = i + 1
    return files


  def iterbatches(self,
                  use_contact_prob=False,
                  batch_size=None,
                  deterministic=True,
                  pad_batches=False,
                  save_path=None):
    """Object that iterates over minibatches from the dataset.
    """
    assert self.X_built, "Dataset not ready for training, features must be built"
    assert self.twoD_X_built, "Dataset not ready for training, 2D features must be built"
    assert self.y_built, "Dataset not ready for training, labels must be built"
    assert self.oneD_y_built, "Dataset not ready for training, 1D labels must be built"
    if use_contact_prob:
      assert self.contact_prob_built
    def iterate(dataset, use_contact_prob, batch_size, deterministic=True, pad_batches=False, on_disk=True):
      n_samples = dataset.get_num_samples()
      if not deterministic:
        sample_perm = np.random.permutation(n_samples)
      else:
        sample_perm = np.arange(n_samples)
      if batch_size is None:
        batch_size = n_samples
      interval_points = np.arange(np.ceil(float(n_samples) / batch_size) + 1) * batch_size
      interval_points[-1] = n_samples
      if on_disk:
        current_end = [0, 0, 0, 0, 0, 0, 0]
        current_start = [0, 0, 0, 0, 0, 0, 0]
        current_files = [0, 0, 0, 0, 0, 0, 0]
        file_sizes = [1000, 100, 100, 100, 500, 500, 100]
        assert batch_size < min(file_sizes)
        X_all = []
        twoD_X_all = []
        y_all = []
        w_all = []
        oneD_y_all = []
        oneD_w_all = []
        if use_contact_prob:
          contact_prob_all = []
        if not deterministic:
          if use_contact_prob:
            X, twoD_X, y, w, oneD_y, oneD_w, contact_prob = SequenceDataset.reorder_Xyw(sample_perm, dataset, path=save_path, use_contact_prob=use_contact_prob)
          else:
            X, twoD_X, y, w, oneD_y, oneD_w = SequenceDataset.reorder_Xyw(sample_perm, dataset, path=save_path)
          sample_perm = np.arange(n_samples)
          deterministic = True
        else:
          X, twoD_X, y, w, oneD_y, oneD_w = dataset.X, dataset.twoD_X, dataset.y, dataset.w, dataset.oneD_y, dataset.oneD_w
          if use_contact_prob:
            contact_prob = dataset.contact_prob

      for j in range(len(interval_points) - 1):
        indices = range(int(interval_points[j]), int(interval_points[j + 1]))
        perm_indices = sample_perm[indices]
        if not on_disk:
          out_X = [dataset.X[i] for i in perm_indices]
          out_twoD_X = [dataset.twoD_X[i] for i in perm_indices]
          out_y = [dataset.y[i] for i in perm_indices]
          out_w = [dataset.w[i] for i in perm_indices]
          out_oneD_y = [dataset.oneD_y[i] for i in perm_indices]
          out_oneD_w = [dataset.oneD_w[i] for i in perm_indices]
          
        else:
          assert deterministic, 'Only support index order'
          index_end = max(perm_indices)
          # X
          while current_end[0] <= index_end:
            X_new = pnet.utils.load_from_joblib(X[current_files[0]])
            current_files[0] = current_files[0] + 1
            current_end[0] = current_end[0] + len(X_new)
            X_all.extend(X_new)
            if len(X_all) > 2*file_sizes[0]:
              del X_all[0:file_sizes[0]]
              current_start[0] = current_start[0] + file_sizes[0]
          # twoD_X
          while current_end[1] <= index_end:
            twoD_X_new = pnet.utils.load_from_joblib(twoD_X[current_files[1]])
            current_files[1] = current_files[1] + 1
            current_end[1] = current_end[1] + len(twoD_X_new)
            twoD_X_all.extend(twoD_X_new)
            if len(twoD_X_all) > 2*file_sizes[1]:
              del twoD_X_all[0:file_sizes[1]]
              current_start[1] = current_start[1] + file_sizes[1]
          # y
          while current_end[2] <= index_end:
            y_new = pnet.utils.load_from_joblib(y[current_files[2]])
            current_files[2] = current_files[2] + 1
            current_end[2] = current_end[2] + len(y_new)
            y_all.extend(y_new)
            if len(y_all) > 2*file_sizes[2]:
              del y_all[0:file_sizes[2]]
              current_start[2] = current_start[2] + file_sizes[2]
          # w
          while current_end[3] <= index_end:
            w_new = pnet.utils.load_from_joblib(w[current_files[3]])
            current_files[3] = current_files[3] + 1
            current_end[3] = current_end[3] + len(w_new)
            w_all.extend(w_new)
            if len(w_all) > 2*file_sizes[3]:
              del w_all[0:file_sizes[3]]
              current_start[3] = current_start[3] + file_sizes[3]
          # oneD_y
          while current_end[4] <= index_end:
            oneD_y_new = pnet.utils.load_from_joblib(oneD_y[current_files[4]])
            current_files[4] = current_files[4] + 1
            current_end[4] = current_end[4] + len(oneD_y_new)
            oneD_y_all.extend(oneD_y_new)
            if len(oneD_y_all) > 2*file_sizes[4]:
              del oneD_y_all[0:file_sizes[4]]
              current_start[4] = current_start[4] + file_sizes[4]
          # oneD_w
          while current_end[5] <= index_end:
            oneD_w_new = pnet.utils.load_from_joblib(oneD_w[current_files[5]])
            current_files[5] = current_files[5] + 1
            current_end[5] = current_end[5] + len(oneD_w_new)
            oneD_w_all.extend(oneD_w_new)
            if len(oneD_w_all) > 2*file_sizes[5]:
              del oneD_w_all[0:file_sizes[5]]
              current_start[5] = current_start[5] + file_sizes[5]
          # contact_prob
          if use_contact_prob:
            while current_end[6] <= index_end:
              contact_prob_new = pnet.utils.load_from_joblib(contact_prob[current_files[6]])
              current_files[6] = current_files[6] + 1
              current_end[6] = current_end[6] + len(contact_prob_new)
              contact_prob_all.extend(contact_prob_new)
              if len(contact_prob_all) > 2*file_sizes[6]:
                del contact_prob_all[0:file_sizes[6]]
                current_start[6] = current_start[6] + file_sizes[6]
              
          out_X = [X_all[i - current_start[0]] for i in perm_indices]
          out_twoD_X = [twoD_X_all[i - current_start[1]] for i in perm_indices]
          out_y = [y_all[i - current_start[2]] for i in perm_indices]
          out_w = [w_all[i - current_start[3]] for i in perm_indices]
          out_oneD_y = [oneD_y_all[i - current_start[4]] for i in perm_indices]
          out_oneD_w = [oneD_w_all[i - current_start[5]] for i in perm_indices]
          if use_contact_prob:
            out_contact_prob = [contact_prob_all[i - current_start[6]] for i in perm_indices]
            
        if pad_batches:
          if not use_contact_prob:
            out_contact_prob = None
          out_X, out_twoD_X, out_y, out_w, out_oneD_y, out_oneD_w, out_contact_prob = dataset.pad_batch(
              batch_size, dataset.n_features, out_X, out_twoD_X, out_y, out_w, out_oneD_y, out_oneD_w, out_contact_prob)
        if use_contact_prob:
          yield out_X, out_twoD_X, out_y, out_w, out_oneD_y, out_oneD_w, out_contact_prob
        else:
          yield out_X, out_twoD_X, out_y, out_w, out_oneD_y, out_oneD_w
    assert self.X_on_disk == self.y_on_disk, 'Only support X, y, w all in memory or in disk'
    return iterate(self, use_contact_prob, batch_size, deterministic, pad_batches, on_disk=self.X_on_disk)

  @staticmethod
  def pad_batch(batch_size, n_features, X, twoD_X, y, w, oneD_y, oneD_w, contact_prob=None):
    """Pads batch to have size of batch_size
    """
    num_samples = len(X)
    assert num_samples <= batch_size
    if num_samples < batch_size:
      X.extend([np.zeros((2, n_features))] * (batch_size - num_samples))
      twoD_X.extend([np.zeros((2, 2, 4))] * (batch_size - num_samples))
      if isinstance(y[0][0, 0], bool):
        y.extend([np.array(np.zeros((2, 2)), dtype=bool)] * (batch_size - num_samples))
      else:
        y.extend([np.zeros((2, 2))] * (batch_size - num_samples))
      w.extend([np.zeros((2, 2))] * (batch_size - num_samples))
      
      oneD_y.extend([np.zeros((2, 2))] * (batch_size - num_samples))
      oneD_w.extend([np.zeros((2,))] * (batch_size - num_samples))
      if contact_prob is not None:
        contact_prob.extend([np.zeros((2, 2))] * (batch_size - num_samples))
    
    return X, twoD_X, y, w, oneD_y, oneD_w, contact_prob

  @staticmethod
  def reorder_Xyw(sample_perm, dataset, path=None, use_contact_prob=False):
    """ Reordering the X, y, w in the dataset according to sample_perm
    """
    assert dataset.X_on_disk
    assert dataset.y_on_disk
    if path is None:
      path = tempfile.mkdtemp()
      paths = [os.path.join(path, 'X'),
               os.path.join(path, 'twoD_X'),
               os.path.join(path, 'y'),
               os.path.join(path, 'w'),
               os.path.join(path, 'oneD_y'),
               os.path.join(path, 'oneD_w')]
      if use_contact_prob:
        paths.append(os.path.join(path, 'contact_prob'))
    targets = [dataset.X, dataset.twoD_X, dataset.y, dataset.w, dataset.oneD_y, dataset.oneD_w]
    if use_contact_prob:
      targets.append(dataset.contact_prob)
    for i, original_paths in enumerate(targets):
      index_order = []
      file_size = []
      for orig_path in original_paths:
        index_order.extend(pnet.utils.load_from_joblib(orig_path))
        file_size.append(len(index_order))
      assert file_size[-1] == dataset.get_num_samples()
      new_order = [index_order[j] for j in sample_perm]
      print("Reordering to %s" % paths[i])
      dataset.save_joblib(new_order, paths[i], file_size=file_size[0])
    if use_contact_prob:
      return dataset.load_joblib(paths[0]), \
             dataset.load_joblib(paths[1]), \
             dataset.load_joblib(paths[2]), \
             dataset.load_joblib(paths[3]), \
             dataset.load_joblib(paths[4]), \
             dataset.load_joblib(paths[5]), \
             dataset.load_joblib(paths[6])
    return dataset.load_joblib(paths[0]), \
           dataset.load_joblib(paths[1]), \
           dataset.load_joblib(paths[2]), \
           dataset.load_joblib(paths[3]), \
           dataset.load_joblib(paths[4]), \
           dataset.load_joblib(paths[5])

  def pick_Xyw(self, sample_perm, data_list, path=None, identifier='X'):
    """ Reordering the X, y, w in the dataset according to sample_perm
    """
    if path is None:
      path = tempfile.mkdtemp()
      path = os.path.join(path, identifier)
    index_order = []
    file_size = []
    for orig_path in data_list:
      index_order.extend(pnet.utils.load_from_joblib(orig_path))
      file_size.append(len(index_order))
    new_order = [index_order[j] for j in sample_perm]
    self.save_joblib(new_order, path, file_size=file_size[0])
    return self.load_joblib(path)

  def itersamples(self, use_contact_prob=False):
    """Object that iterates over the samples in the dataset.
    """
    assert self.X_built, "Dataset not ready for training, features must be built"
    assert self.y_built, "Dataset not ready for training, labels must be built"
    def sample_iterate(dataset, use_contact_prob=False, on_disk=True):
      n_samples = dataset.get_num_samples()
      if on_disk:
        current_X = 0
        current_twoD_X = 0
        current_y = 0
        current_w = 0
        current_oneD_y = 0
        current_oneD_w = 0
        current_X_file = 0
        current_twoD_X_file = 0
        current_y_file = 0
        current_w_file = 0
        current_oneD_y_file = 0
        current_oneD_w_file = 0
        if use_contact_prob:
          current_contact_prob = 0
          current_contact_prob_file = 0
        X_all = []
        twoD_X_all = []
        y_all = []
        w_all = []
        oneD_y_all = []
        oneD_w_all = []
        if use_contact_prob:
          contact_prob_all = []

      for i in range(n_samples):
        if not on_disk:
          yield self.X[i], self.twoD_X[i], self.y[i], self.w[i], self.oneD_y[i], self.oneD_w[i]
        else:
          while current_X <= i:
            X_new = pnet.utils.load_from_joblib(self.X[current_X_file])
            current_X_file = current_X_file + 1
            current_X = current_X + len(X_new)
            X_all.extend(X_new)
          while current_twoD_X <= i:
            twoD_X_new = pnet.utils.load_from_joblib(self.twoD_X[current_twoD_X_file])
            current_twoD_X_file = current_twoD_X_file + 1
            current_twoD_X = current_twoD_X + len(twoD_X_new)
            twoD_X_all.extend(twoD_X_new)
          while current_y <= i:
            y_new = pnet.utils.load_from_joblib(self.y[current_y_file])
            current_y_file = current_y_file + 1
            current_y = current_y + len(y_new)
            y_all.extend(y_new)
          while current_w <= i:
            w_new = pnet.utils.load_from_joblib(self.w[current_w_file])
            current_w_file = current_w_file + 1
            current_w = current_w + len(w_new)
            w_all.extend(w_new)
          while current_oneD_y <= i:
            oneD_y_new = pnet.utils.load_from_joblib(self.oneD_y[current_oneD_y_file])
            current_oneD_y_file = current_oneD_y_file + 1
            current_oneD_y = current_oneD_y + len(oneD_y_new)
            oneD_y_all.extend(oneD_y_new)
          while current_oneD_w <= i:
            oneD_w_new = pnet.utils.load_from_joblib(self.oneD_w[current_oneD_w_file])
            current_oneD_w_file = current_oneD_w_file + 1
            current_oneD_w = current_oneD_w + len(oneD_w_new)
            oneD_w_all.extend(oneD_w_new)
          if use_contact_prob:
            while current_contact_prob <= i:
              contact_prob_new = pnet.utils.load_from_joblib(self.contact_prob[current_contact_prob_file])
              current_contact_prob_file = current_contact_prob_file + 1
              current_contact_prob = current_contact_prob + len(contact_prob_new)
              contact_prob_all.extend(contact_prob_new)
            yield X_all[i], twoD_X_all[i], y_all[i], w_all[i], oneD_y_all[i], oneD_w_all[i], contact_prob_all[i]
          else:
            yield X_all[i], twoD_X_all[i], y_all[i], w_all[i], oneD_y_all[i], oneD_w_all[i]
    assert self.X_on_disk == self.y_on_disk, 'Only support X, y, w all in memory or in disk'
    return sample_iterate(self, use_contact_prob=use_contact_prob, on_disk=self.X_on_disk)

  def train_valid_test_split(self,
                             deterministic=False,
                             train_frac=0.8,
                             valid_frac=0.1,
                             test_frac=0.1):
    n_samples = self.get_num_samples()
    if not deterministic:
      sample_perm = np.random.permutation(n_samples)
    else:
      sample_perm = np.arange(n_samples)
    assert np.allclose(train_frac + valid_frac + test_frac, 1)
    train = self.select_by_index(sample_perm[:int(train_frac*n_samples)])
    valid = self.select_by_index(sample_perm[int(train_frac*n_samples):int((train_frac+valid_frac)*n_samples)])
    test = self.select_by_index(sample_perm[int((train_frac+valid_frac)*n_samples):])
    return train, valid, test