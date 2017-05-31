#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:19:12 2017

@author: zqwu
"""

import numpy as np
import os
import Bio
import tempfile
import pnet

from Bio.Blast.Applications import NcbiblastpCommandline, NcbipsiblastCommandline
from Bio.Blast import NCBIXML, NCBIWWW

def blastp_remote(seq, database='nr'):\
  result = Bio.Blast.NCBIWWW.qblast("blastp", database, seq)
  blast_records = NCBIXML.parse(result)
  
def blastp_local(seq, database='nr', data_dir=None):
  save_dir = tempfile.mkdtemp()
  pnet.utils.write_sequence(seq, os.path.join(save_dir, 'temp.seq'))
  if data_dir is None:
    data_dir = os.environ['BLAST_DATA_DIR']
  cline = NcbiblastpCommandline(query=os.path.join(save_dir, 'temp.seq'), 
                                db=os.path.join(data_dir, database), 
                                out=os.path.join(save_dir, 'blastp_results.xml'),
                                evalue=0.001,
                                outfmt=5)
  stdout, stderr = cline()
  with open(os.path.join(save_dir, 'blastp_results.xml'), 'r') as f:
    blast_records = list(NCBIXML.parse(f))
    
  
  
def psiblast_local(seq, database='nr', data_dir=None):
  save_dir = tempfile.mkdtemp()
  pnet.utils.write_sequence(seq, os.path.join(save_dir, 'temp.seq'))
  if data_dir is None:
    data_dir = os.environ['BLAST_DATA_DIR']
  cline = NcbipsiblastCommandline(query=os.path.join(save_dir, 'temp.seq'), 
                                  db=os.path.join(data_dir, database), 
                                  out=os.path.join(save_dir, 'psiblast_results.xml'),
                                  evalue=0.001,
                                  num_iterations=2,
                                  out_pssm=os.path.join(save_dir, 'psiblast_results.pssm'),
                                  save_each_pssm=True,
                                  outfmt=5)
  stdout, stderr = cline()
  with open(os.path.join(save_dir, 'psiblast_results.xml'), 'r') as f:
    blast_records = list(NCBIXML.parse(f))
  
def system_call(command):
  p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
  return p.stdout.read()