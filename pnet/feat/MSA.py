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
import Bio.SeqIO
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

def generate_msa(dataset, mode="hhblits", evalue=0.001, num_iterations=3, reload=True):
  """ Load multiple sequence alignment features(position frequency matrix) """
  msa_path = form_msa(dataset, mode=mode,
                      evalue=evalue,
                      num_iterations=num_iterations,
                      reload=reload)
  with open(msa_path, 'r') as f:
    sequences = []
    for record in Bio.SeqIO.parse(f, 'fasta'):
      seq = []
      for res in record.seq.tostring():
        try:
          seq.append(AminoAcid[res.upper()])
        except:
          # Unknown Amino Acid
          seq.append(None)
      sequences.append(seq)

  with open(msa_path, 'r') as f:
    for first_record in Bio.SeqIO.parse(f, 'fasta'):
      break
  # Valid residue index
  index = [i for i, res in enumerate(first_record.seq.tostring()) if res != '-']
  sequences = np.transpose(np.array(sequences))[index, :]
  def to_one_hot(res):
    result = np.zeros((25))
    if res >= 0:
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

def form_msa(dataset, mode="hhblits", evalue=0.001, num_iterations=3, reload=True):
  """Generate multiple sequence alignment for a single sample(sequence)"""
  assert len(dataset.sequences) == 1, "Only support one sample"
  # Support psiblast, blastp
  data_dir = os.environ['PNET_DATA_DIR']
  msa_file = os.path.join(data_dir, 'MSA_ALL/'+dataset.IDs[0]+'.fas')
  if os.path.exists(msa_file) and reload:
    return msa_file
  if mode == "psiblast":
    path, e = psiblast_local(dataset, evalue=evalue, num_iterations=num_iterations)
  elif mode == "blastp":
    path, e = blastp_local(dataset, evalue=evalue)
  elif mode == "hhblits":
    _, path = hhblits_local(dataset, evalue=evalue, num_iterations=num_iterations)
    e = None
  return path
