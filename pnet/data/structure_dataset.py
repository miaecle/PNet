#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:52:05 2017

@author: zqwu
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from pnet.data import SequenceDataset

class StructureDataset(SequenceDataset):
  def __init__(self, structures, resseqs, IDs, **kwargs):
    self._structures = structures
    self._resseqs = np.array(resseqs) - 1
    self.xyz = []
    for i, structure in enumerate(self._structures):
      if structure is None:
        self.xyz.append(None)
      else:
        coordinate = np.zeros((max(self._resseqs[i]) + 1,3))
        coordinate[self._resseqs[i], :] = structure.xyz
        self.xyz.append(coordinate)
    super(StructureDataset, self).__init__(IDs, **kwargs)

  @staticmethod
  def from_SequenceDataset(structures, resseqs, dataset):
    IDs = dataset._IDs
    sequences = dataset._sequences
    pdb_paths = dataset._pdb_paths
    raw = dataset._raw
    return StructureDataset(structures, resseqs, IDs, sequences=sequences,
                            pdb_paths=pdb_paths, raw=raw)

  @property
  def structures(self):
    """ Get structures as np.array (of mdtraj.Trajectory object) """
    return np.array(self._structures)

  @property
  def resseqs(self):
    """ Get residue index of all CB atoms """
    return np.array(self._resseqs)

  def select_by_index(self, index):
    """Creates a new dataset from a selection of index.
    """
    IDs = self.IDs[index]
    sequences = self.sequences[index]
    pdb_paths = self.pdb_paths[index]
    raw = self.raw[index]
    msas = self.msas[index]
    structures = self.structures[index]
    resseqs = self.resseqs[index]
    return StructureDataset(structures, resseqs, IDs, sequences, pdb_paths, raw, msas)

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
    msas_add = list(np.array(extension_dataset.msas)[add_on])
    structures_add = list(np.array(extension_dataset.structures)[add_on])
    resseqs_add = list(np.array(extension_dataset.resseqs)[add_on])
    self._IDs.extend(IDs_add)
    self._pdb_paths.extend(pdb_paths_add)
    self._sequences.extend(sequences_add)
    self._raw.extend(raw_add)
    self._msas.extend(msas_add)
    self._structures.extend(structures_add)
    self._resseqs.extend(resseqs_add)
    self.n_samples = self.n_samples + len(add_on)

  def build_residue_contact(self, binary=False, threshold=0.8):
    """ Build residue-residue contact map based on 3D coordinates"""
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
    return self.RRs