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
from pnet.utils.amino_acids import AminoAcid_Entropy, contact_potential
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
    mats.append(np.expand_dims(load_psicov_mat(psicov_file), axis=2))
  return mats

def load_psicov_mat(path):
  mat = np.array(pd.read_table(path, header=None))
  n_res = mat.shape[0]
  return mat[:n_res, :n_res]

def MI_MCP(dataset):
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
  
  out = []
  for i in range(dataset.n_samples):  
    MI_file = data_dir+'/MI_ALL/'+dataset._IDs[i]+'_mi_mcp.joblib'
    if os.path.exists(MI_file):
      dat = joblib.load(MI_file)
      MI = dat[0]
      MCP = dat[1]
      out.append(np.concatenate([MI, MCP], axis=2))
    else:
      try:
        msa_path = form_msa(dataset.select_by_index(i))
        with open(msa_path, 'r') as f:
          sequences = [[MapToID(res.upper()) for res in record.seq.tostring()] for record in Bio.SeqIO.parse(f, 'fasta')]
        with open(msa_path, 'r') as f:
          for first_record in Bio.SeqIO.parse(f, 'fasta'):
            break
        index = [j for j, res in enumerate(first_record.seq.tostring()) if res != '-']
        sequences = np.array(sequences)[:, index]
      
        num_res = len(index)
        feed_dict = {n_res: num_res}
        out_ct_1D = np.zeros((num_res, 21))
        out_ct_2D = np.zeros((num_res, num_res, 21, 21))
        
        batch_size = int(5*(700/num_res)**2)
        n_batch = sequences.shape[0]//batch_size+1
        print(dataset._IDs[i] +': ' + str(num_res) + ', ' + str(sequences.shape[0]))
        for j in range(n_batch):
          if j==n_batch-1:
            feed_dict[inputs] = sequences[j*batch_size:, :]
          else:
            feed_dict[inputs] = sequences[j*batch_size:(j+1)*batch_size, :]
          out_ct_1D += sess.run(ct_1D, feed_dict=feed_dict)
          out_ct_2D += sess.run(ct_2D, feed_dict=feed_dict)
        freq_1D = out_ct_1D/sequences.shape[0]
        freq_2D = np.reshape(out_ct_2D, (num_res, num_res, 21*21))/sequences.shape[0]
        
        MI = mutual_information(freq_1D, freq_2D)
        MCP = mean_contact_potential(freq_1D, freq_2D)
        joblib.dump([MI, MCP], MI_file)
        out.append(np.concatenate([MI, MCP], axis=2))
      except:
        continue
  return out
  
def mutual_information(freq_1D, freq_2D):
  num_res = freq_1D.shape[0]
  H_1D = np.sum(-freq_1D * np.log(freq_1D + 1e-7) / np.log(21), axis=1)
  H_2D = np.sum(-freq_2D * np.log(freq_2D + 1e-7) / np.log(21), axis=2)
      
  MI = -H_2D + np.stack([H_1D]*num_res, axis=1) + np.stack([H_1D]*num_res, axis=0)
      
  MI_1D = (np.sum(MI, axis=1) - np.diag(MI))/(num_res-1)
  MI_av = (np.sum(MI) - np.sum(np.diag(MI)))/num_res/(num_res-1)
  APC = np.stack([MI_1D]*num_res, axis=1)*np.stack([MI_1D]*num_res, axis=0)/MI_av
  normalized_MI = MI - APC
  return np.stack([MI, normalized_MI], axis=2)

def mean_contact_potential(freq_1D, freq_2D):
  mcp = np.sum(freq_2D*np.expand_dims(np.expand_dims(contact_potential, axis=0), axis=0), axis=2, keepdims=True)
  return mcp
  
