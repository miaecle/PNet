#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:15:59 2017

@author: zqwu
"""
import os
import pandas as pd
import numpy as np
import csv
from pnet.utils import blastp_local, psiblast_local, hhblits_local
from pnet.utils.amino_acids import AminoAcid

def generate_raw(dataset):
  """ Generate one hot feature for each residue """
  assert len(dataset.sequences) == 1
  sequences = [AminoAcid[res] for res in dataset.sequences[0]]
  return np.reshape(to_one_hot(np.array(sequences), n_classes=25), (len(sequences), 25))

def to_one_hot(y, n_classes=25):
  """ Sparse to one hot """
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, n_classes))
  y_hot[np.arange(n_samples), y.astype(np.int64)] = 1
  return y_hot

def generate_msa(dataset, mode="hhblits", evalue=0.001, num_iterations=2, reload=True):
  """ Load multiple sequence alignment features(position frequency matrix) """
  msa = form_msa(dataset, mode=mode,
                 evalue=evalue,
                 num_iterations=num_iterations,
                 reload=reload)
  sequences = [[AminoAcid[res.upper()] for res in msa._hit_sequences[i]] for i in range(len(msa._hit_sequences))]
  # Valid residue index
  index = [i for i, res in enumerate(msa.master_sequence) if res != '-']
  sequences = np.transpose(np.array(sequences))[index, :]
  def to_one_hot(res):
    result = np.zeros((25))
    if res > 0:
      result[int(res)] = 1
    return result
  sequences_one_hot = [[to_one_hot(sequence[i]) for i in range(len(sequence))] for sequence in sequences]
  pfm = np.sum(np.array(sequences_one_hot), axis=1)
  for i, res_freq in enumerate(pfm):
    total_count = np.sum(res_freq)
    if total_count > 0:
      # Calculate frequency
      pfm[i, :] = pfm[i, :]/total_count
  return pfm

def form_msa(dataset, mode="psiblast", evalue=0.001, num_iterations=2, reload=True):
  """Generate multiple sequence alignment for a single sample(sequence)"""
  assert len(dataset.sequences) == 1, "Only support one sample"
  # Support psiblast, blastp
  data_dir = os.environ['PNET_DATA_DIR']
  msa_file = os.path.join(data_dir, 'MSA_All/'+dataset.IDs[0]+'.msa')
  if os.path.exists(msa_file) and reload:
    # Load from file
    return load_msa(msa_file)
  if mode == "psiblast":
    path, e = psiblast_local(dataset, evalue=evalue, num_iterations=num_iterations)
  elif mode == "blastp":
    path, e = blastp_local(dataset, evalue=evalue)
  elif mode == "hhblits":
    path, path2 = hhblits_local(dataset, evalue=evalue, num_iterations=num_iterations)
    e = None
  msa = load_msa_from_aln(path, e=e)
  if reload:
    write_msa(msa, msa_file)
  return load_msa_from_aln(path, e=e)

def load_msa_from_aln(path, e=None):
  """ ClustalW .aln file loader """
  with open(path, 'r') as f:
    raw = f.readlines()
  lines = [line.split() for line in raw]
  #lengths = map(len, lines)
  start_pos = []
  end_pos = []
  start = False
  for i, line in enumerate(raw[1:]):
    if line[0] != '\n' and line[0] != ' ' and start == False:
      start_pos.append(i+1)
      start = True
    if (line[0] == '\n' or line[0] == ' ') and start == True:
      end_pos.append(i+1)
      start = False
  num_hits = np.unique(np.array(end_pos)-np.array(start_pos))
  assert len(num_hits) == 1
  num_blocks = len(start_pos)
  sequences = ['']*num_hits[0]
  IDs = list(np.array(lines[start_pos[0]:end_pos[0]])[:, 0])
  for i in range(num_blocks):
    seq_add = np.array(lines[start_pos[i]:end_pos[i]])[:, 1]
    for j in range(len(seq_add)):
      sequences[j] = sequences[j] + seq_add[j]
  return MultipleSequenceAlignment(IDs, sequences, e=e, path=path)

def load_msa(path):
  """ Load msa(customized class) from file """
  df = pd.read_csv(path, header=None)
  IDs = df[0].tolist()
  sequences = df[1].tolist()
  if len(df.columns) == 3:
    e = df[2].tolist()
    e = e[1:]
  else:
    e = None
  return MultipleSequenceAlignment(IDs, sequences, e=e, path=None)

def write_msa(msa_dataset, path):
  """ Write msa(customized class) to file """
  IDs = msa_dataset.hit_IDs
  sequences = msa_dataset.hit_sequences
  e = msa_dataset.e
  n_hits = len(IDs)
  if len(e)>0 and max(e) > 0:
    write_e = True
    e = np.insert(e, 0, 0)
  else:
    write_e = False
  with open(path, 'w') as f:
    writer = csv.writer(f)
    for i in range(n_hits):
      out_line = [IDs[i], sequences[i]]
      if write_e:
        out_line.append(e[i])
      writer.writerow(out_line)

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