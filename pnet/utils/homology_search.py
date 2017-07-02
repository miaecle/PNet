#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:19:12 2017

@author: zqwu
"""

import numpy as np
import os
import pandas as pd
import tempfile
import pnet
import subprocess

from Bio.Blast.Applications import NcbiblastpCommandline, NcbipsiblastCommandline

def system_call(command):
  """ Wrapper for system command call """
  p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
  return p.stdout.read()

def blastp_local(dataset, database='nr', data_dir=None, evalue=0.001):
  save_dir = tempfile.mkdtemp()
  pnet.utils.write_dataset(dataset, os.path.join(save_dir, 'temp.seq'))
  if data_dir is None:
    data_dir = os.environ['BLAST_DATA_DIR']
  cline = NcbiblastpCommandline(query=os.path.join(save_dir, 'temp.seq'),
                                db=os.path.join(data_dir, database),
                                out=os.path.join(save_dir, 'blastp_results.csv'),
                                evalue=evalue,
                                max_target_seqs=1000,
                                outfmt=10)
  stdout, stderr = cline()
  df = pd.read_csv(os.path.join(save_dir, 'psiblast_results.csv'), header=None)
  hit_IDs = [hit_ID for hit_ID in df[1] if not pd.isnull(hit_ID)]
  hit_IDs, indices = np.unique(hit_IDs, return_index=True)
  hit_e = np.array(df[10].tolist())[indices]
  with open(os.path.join(save_dir, 'hits.csv'), 'w') as f:
    for i in range(len(hit_IDs)):
      f.write(hit_IDs[i]+'\n')
  command = 'blastdbcmd -out ' + os.path.join(save_dir, 'hits.fas') + \
            ' -db ' + os.path.join(data_dir, database) + \
            ' -entry_batch ' + os.path.join(save_dir, 'hits.csv')
  flag = system_call(command)
  with open(os.path.join(save_dir, 'hits.fas'), 'r') as f:
    hits_sequences = f.readlines()
  with open(os.path.join(save_dir, 'temp.seq'), 'a') as f:
    f.writelines(hits_sequences)
  command = 'clustalw -infile=' + os.path.join(save_dir, 'temp.seq') + \
            ' -outfile=' + os.path.join(save_dir, 'results.aln') + \
            ' -OUTORDER=INPUT'
  flag = system_call(command)
  return os.path.join(save_dir, 'results.aln'), hit_e

def psiblast_local(dataset, database='nr', data_dir=None, evalue=0.001, num_iterations=2):
  save_dir = tempfile.mkdtemp()
  pnet.utils.write_dataset(dataset, os.path.join(save_dir, 'temp.seq'))
  if data_dir is None:
    data_dir = os.environ['BLAST_DATA_DIR']
  cline = NcbipsiblastCommandline(query=os.path.join(save_dir, 'temp.seq'),
                                  db=os.path.join(data_dir, database),
                                  out=os.path.join(save_dir, 'psiblast_results.csv'),
                                  evalue=evalue,
                                  num_iterations=num_iterations,
                                  out_ascii_pssm=os.path.join(save_dir, 'results.pssm'),
                                  save_pssm_after_last_round=True,
                                  max_target_seqs=1000,
                                  outfmt=10)
  stdout, stderr = cline()
  df = pd.read_csv(os.path.join(save_dir, 'psiblast_results.csv'), header=None)
  hit_IDs = [hit_ID for hit_ID in df[1] if not pd.isnull(hit_ID)]
  hit_IDs, indices = np.unique(hit_IDs, return_index=True)
  hit_e = np.array(df[10].tolist())[indices]
  with open(os.path.join(save_dir, 'hits.csv'), 'w') as f:
    for i in range(len(hit_IDs)):
      f.write(hit_IDs[i]+'\n')
  command = 'blastdbcmd -out ' + os.path.join(save_dir, 'hits.fas') + \
            ' -db ' + os.path.join(data_dir, database) + \
            ' -entry_batch ' + os.path.join(save_dir, 'hits.csv')
  flag = system_call(command)
  with open(os.path.join(save_dir, 'hits.fas'), 'r') as f:
    hits_sequences = f.readlines()
  with open(os.path.join(save_dir, 'temp.seq'), 'a') as f:
    f.writelines(hits_sequences)
  command = 'clustalw -infile=' + os.path.join(save_dir, 'temp.seq') + \
            ' -outfile=' + os.path.join(save_dir, 'results.aln') + \
            ' -OUTORDER=INPUT'
  flag = system_call(command)
  return os.path.join(save_dir, 'results.aln'), hit_e

def hhblits_local(dataset, database='uniprot20_2016_02', data_dir=None,
                  evalue=0.001, num_iterations=2, num_threads=4):
  save_dir = tempfile.mkdtemp()
  pnet.utils.write_dataset(dataset, os.path.join(save_dir, 'temp.seq'))
  if data_dir is None:
    data_dir = os.environ['HH_DATA_DIR']
  command = 'hhblits -i ' + os.path.join(save_dir, 'temp.seq') + \
            ' -d ' + os.path.join(data_dir, database) + \
            ' -oa3m ' + os.path.join(save_dir, 'results.a3m') + \
            ' - cpu ' + str(num_threads) + ' -n ' + str(num_iterations) + \
            ' -e ' + str(evalue)
  flag = system_call(command)
  command = 'reformat.pl a3m clu ' + os.path.join(save_dir, 'results.a3m') + \
            ' ' + os.path.join(save_dir, 'results.clu')
  flag = system_call(command)
  return os.path.join(save_dir, 'results.clu')