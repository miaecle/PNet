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

def pad_batch(batch_size, dataset):
  """Pads batch to have size of batch_size
  """
  num_samples = dataset.get_num_samples()
  assert num_samples <= batch_size
  if num_samples < batch_size:
    dataset._IDs.extend([None] * (batch_size - num_samples))
    dataset._sequences.extend([None] * (batch_size - num_samples))
    dataset._pdb_paths.extend([None] * (batch_size - num_samples))
    dataset._raw.extend([None] * (batch_size - num_samples))
    dataset._msas.extend([None] * (batch_size - num_samples))
    try:
      dataset._structures.extend([None] * (batch_size - num_samples))
      dataset._resseqs.extend([None] * (batch_size - num_samples))
      dataset.xyz.extend([None] * (batch_size - num_samples))
    except:
      pass
  return dataset

def merge_datasets(datasets):
  dataset = datasets[0]
  for i in range(1, len(datasets)):
    dataset.extend(datasets[i])
  return dataset

class SequenceDataset(object):
  """
  Dataset class for protein sequences
  """
  def __init__(self, IDs, sequences=None, pdb_paths=None, raw=None, msas=None):
    """Hold information of IDs, sequences, path of pdb files and raw inputs"""
    self._IDs = list(IDs)
    self.n_samples = len(IDs)
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
    if msas is None:
      self._msas = [None] * self.n_samples
    else:
      self._msas = list(msas)

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
    msas_add = list(np.array(extension_dataset.msas)[add_on])
    self._IDs.extend(IDs_add)
    self._pdb_paths.extend(pdb_paths_add)
    self._sequences.extend(sequences_add)
    self._raw.extend(raw_add)
    self._msas.extend(msas_add)
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
  def msas(self):
    """ Get all multiple sequence alignments as np.array"""
    return np.array(self._msas)

  def iterbatches(self,
                  batch_size=None,
                  deterministic=False,
                  pad_batches=False):
    """Object that iterates over minibatches from the dataset.
    """
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
        out_dataset = dataset.select_by_index(perm_indices)
        if pad_batches:
          out_dataset = pad_batch(batch_size, out_dataset)
        yield out_dataset
    return iterate(self, batch_size, deterministic, pad_batches)

  def itersamples(self):
    """Object that iterates over the samples in the dataset.
    """
    def sample_iterate(dataset):
      n_samples = dataset.get_num_samples()
      for i in range(n_samples):
        yield dataset.select_by_index(i)
    return sample_iterate(self)

  def select_by_index(self, index):
    """Creates a new dataset from a selection of index.
    """
    IDs = self.IDs[index]
    sequences = self.sequences[index]
    pdb_paths = self.pdb_paths[index]
    raw = self.raw[index]
    msas = self.msas[index]
    return SequenceDataset(IDs, sequences, pdb_paths, raw, msas)

  def select_by_ID(self, IDs_selection):
    """Creates a new dataset from a selection of IDs.
    """
    index = []
    for i, ID in enumerate(self._IDs):
      if ID in IDs_selection:
        index.append(i)
    return self.select_by_index(index)

  def __getitem__(self, index):
    return self.select_by_index(index)

class MultipleSequenceAlignment(object):
  """
  class for MSA
  """
  def __init__(self, hit_IDs, hit_sequences, e=None, path=None):
    """Hold information of master sequence(and ID), all hits above evalue"""
    self._hit_IDs = list(hit_IDs)
    self._hit_sequences = list(hit_sequences)
    self.master_ID = self._hit_IDs[0]
    self.master_sequence = self._hit_sequences[0]
    # Number of alignments should exclude master sequence
    self.n_alignments = len(self._hit_IDs) - 1
    self.path = path
    if e is None:
      self._e = [0] * self.n_alignments
    else:
      self._e = list(e)
      assert len(self._e) == self.n_alignments

  @property
  def hit_IDs(self):
    """ Get IDs as np.array"""
    return np.array(self._hit_IDs)

  @property
  def hit_sequences(self):
    """ Get sequences as np.array"""
    return np.array(self._hit_sequences)

  @property
  def e(self):
    """ Get sequences as np.array"""
    return np.array(self._e)