#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:16:17 2017

@author: zqwu
"""

import os
import pandas as pd
import numpy as np
import csv
import Bio.SeqIO
import tempfile
from pnet.utils import blastp_local, psiblast_local, hhblits_local
from pnet.utils.amino_acids import AminoAcid_Entropy
from pnet.feat.MSA import form_msa
import tensorflow as tf
import joblib


def CCMpred(dataset, ccmpred_path='/CCMpred'):
  ccmpred = os.environ['HOME']+ccmpred_path+'/bin/ccmpred'
  preprocess = os.environ['HOME']+ccmpred_path+'/scripts/convert_alignment.py'
  assert os.path.exists(ccmpred), 'can not find CCMpred'
  data_dir = os.environ['PNET_DATA_DIR']
  
  mats = []
  for i in range(dataset.n_samples):
    psicov_file = os.path.join(data_dir, 'PSICOV_ALL/'+dataset.IDs[i]+'.mat')
    if not os.path.exists(psicov_file):
      path = form_msa(dataset.select_by_index(i))
      os.system('python '+preprocess+' '+path+' fasta /tmp/temp.aln')
      os.system(ccmpred + ' /tmp/temp.aln ' + psicov_file)
    mats.append(load_psicov_mat(psicov_file))
  return mats

def load_psicov_mat(path):
  mat = np.array(pd.read_table(path, header=None))
  n_res = mat.shape[0]
  return mat[:n_res, :n_res]

def mutual_information(dataset):
  def MapToID(char):
    if char in AminoAcid_Entropy.keys():
      return AminoAcid_Entropy[char]
    else:
      return 20
  data_dir = os.environ['PNET_DATA_DIR']
  
  g = tf.Graph()
  with g.as_default():
    n_res = tf.placeholder(tf.int32, shape=())
    inputs = tf.placeholder(tf.int32, shape=(None, None)) # n_alignments * length
    count_1D = tf.one_hot(inputs, 21, axis=-1)
    ct_1D = tf.reduce_sum(count_1D, axis=0)

    tensor_1 = tf.tile(tf.expand_dims(tf.expand_dims(count_1D, axis=1), axis=3), (1, n_res, 1, 21, 1))
    tensor_2 = tf.tile(tf.expand_dims(tf.expand_dims(count_1D, axis=2), axis=4), (1, 1, n_res, 1, 21))
    ct_2D = tf.reduce_sum(tensor_1*tensor_2, axis=0)
    sess = tf.Session(graph=g)
  
  for i in range(dataset.n_samples):  
    if os.path.exists(data_dir+'/MI_ALL/'+dataset._IDs[i]+'.mi'):
      continue
    msa_path = form_msa(dataset.select_by_index(i))
    with open(msa_path, 'r') as f:
      sequences = [[MapToID(res.upper()) for res in record.seq.tostring()] for record in Bio.SeqIO.parse(f, 'fasta')]
    with open(msa_path, 'r') as f:
      for first_record in Bio.SeqIO.parse(f, 'fasta'):
        break
    index = [i for i, res in enumerate(first_record.seq.tostring()) if res != '-']
    sequences = np.array(sequences)[:, index]
  
    num_res = len(index)
    feed_dict = {n_res: num_res}
    out_ct_1D = np.zeros((num_res, 21))
    out_ct_2D = np.zeros((num_res, num_res, 21, 21))
  
    n_batch = sequences.shape[0]//20+1
    for j in range(n_batch):
      if j==n_batch-1:
        feed_dict[inputs] = sequences[j*20:, :]
      else:
        feed_dict[inputs] = sequences[j*20:(j+1)*20, :]
      out_ct_1D += sess.run(ct_1D, feed_dict=feed_dict)
      out_ct_2D += sess.run(ct_2D, feed_dict=feed_dict)
    freq_1D = out_ct_1D/sequences.shape[0]
    freq_2D = np.reshape(out_ct_2D, (num_res, num_res, 21*21))/sequences.shape[0]
    
    H_1D = np.sum(-freq_1D * np.log(freq_1D + 1e-7) / np.log(21), axis=1)
    H_2D = np.sum(-freq_2D * np.log(freq_2D + 1e-7) / np.log(21), axis=2)
    
    MI = -H_2D + np.stack([H_1D]*num_res, axis=1) + np.stack([H_1D]*num_res, axis=0)
    
    MI_1D = (np.sum(MI, axis=1) - np.diag(MI))/(num_res-1)
    MI_av = (np.sum(MI) - np.sum(np.diag(MI)))/num_res/(num_res-1)
    APC = np.stack([MI_1D]*num_res, axis=1)*np.stack([MI_1D]*num_res, axis=0)/MI_av
    normalized_MI = MI - APC
    
    joblib.dump([freq_1D, freq_2D, MI, normalized_MI], data_dir+'/MI_ALL/'+dataset._IDs[i]+'.mi')
  


  
