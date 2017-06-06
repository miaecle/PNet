#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:40:00 2017

@author: zqwu
"""
import os
import pandas as pd
import numpy as np
import csv
from pnet.data import SequenceDataset, MultipleSequenceAlignment
from pnet.data import merge_datasets
from pnet.models.homology_search import blastp_local, psiblast_local

def generate_msa(dataset, mode="psiblast", evalue=0.001, num_iterations=2):
  """Generate multiple sequence alignment for a single sample(sequence)"""
  assert len(dataset.sequences) == 1, "Only support one sample"
  # Support psiblast, blastp
  if mode == "psiblast":
    path, e = psiblast_local(dataset, evalue=evalue, num_iterations=num_iterations)
  elif mode == "blastp":
    path, e = blastp_local(dataset, evalue=evalue)
  return load_msa_from_aln(path, e=e)

def load_msa_from_aln(path, e=None):
  """ ClustalW .aln file loader """
  with open(path, 'r') as f:
    lines = f.readlines()
  lines = [line.split() for line in lines]
  lengths = map(len, lines)
  start_pos = []
  end_pos = []
  start = False
  for i, length in enumerate(lengths[1:]):
    if length > 0 and start == False:
      start_pos.append(i+1)
      start = True
    if length == 0 and start == True:
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
  IDs = msa_dataset.IDs
  sequences = msa_dataset.sequences
  e = msa_dataset.e
  n_hits = len(IDs)
  if max(e) > 0:
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
