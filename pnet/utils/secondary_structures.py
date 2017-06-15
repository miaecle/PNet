#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 21:59:30 2017

@author: zqwu
"""

import numpy as np
import os
import pandas as pd
import pnet
from pnet.utils import system_call

def string_to_onehot_ss(ss):
  state = {'C': [1, 0, 0], 'H': [0, 1, 0], 'E': [0, 0, 1]}
  return np.array([state[a] for a in ss])

def read_ss(path, dataset):
  with open(path, 'r') as f:
    lines = f.readlines()
  lines = [line.split() for line in lines]
  start = 0
  length = len(dataset.sequences[0])
  for i, line in enumerate(lines):
    if len(line) == 6 and line[0] == '1' and line[2] in ['C', 'E', 'H']:
      start = i
      break
  data = np.array(lines[start:start+length])
  seq = ''.join(list(data[:,1]))
  assert str(seq) == str(dataset.sequences[0])
  coil_defined = 1
  sheet_defined = 1
  helix_defined = 1
  current_line = 0
  order = np.zeros(3)
  while coil_defined + sheet_defined + helix_defined > 1:
    if data[current_line, 2] == 'C' and coil_defined > 0:
      values = np.array(data[current_line, 3:6])
      a = np.argmax(values)
      if values[a-1] < values[a] and values[a-2] < values[a]:
        order[0] = a + 3
        coil_defined = 0
    elif data[current_line, 2] == 'E' and sheet_defined > 0:
      values = np.array(data[current_line, 3:6])
      a = np.argmax(values)
      if values[a-1] < values[a] and values[a-2] < values[a]:
        order[1] = a + 3
        sheet_defined = 0
    elif data[current_line, 2] == 'H' and helix_defined > 0:
      values = np.array(data[current_line, 3:6])
      a = np.argmax(values)
      if values[a-1] < values[a] and values[a-2] < values[a]:
        order[2] = a + 3
        helix_defined = 0
    if coil_defined + sheet_defined + helix_defined == 1:
      order[np.argmin(order)] = 12 - np.sum(order)
    current_line = current_line + 1
  assert sorted(order) == [3, 4, 5]
  order = np.array(order, dtype=int)
  return np.array(np.stack([data[:, order[0]],
                            data[:, order[1]],
                            data[:, order[2]]]), dtype=float)

def raptorx_ss(dataset):
  raptorx_dir = os.environ['RAPTORX_DIR']
  original_dir = os.getcwd()
  os.chdir(raptorx_dir)
  system_call('rm temp.seq')
  system_call('rm -rf ./tmp/temp/')
  pnet.utils.write_dataset(dataset, os.path.join(raptorx_dir, 'temp.seq'))
  command = './oneline_command.sh temp.seq'
  system_call(command)
  path = os.path.join(raptorx_dir, 'tmp/temp/temp.ss3')
  os.chdir(original_dir)
  return read_ss(path, dataset)

def psspred_ss(dataset):
  psspred_dir = os.environ['PSSPRED_DIR']
  original_dir = os.getcwd()
  os.chdir(psspred_dir)
  system_call('rm -rf ./temp/')
  system_call('mkdir temp')
  os.chdir('./temp')
  pnet.utils.write_dataset(dataset, os.path.join(psspred_dir, 'temp/temp.seq'))
  command = 'perl ../PSSpred.pl temp.seq'
  system_call(command)
  path = os.path.join(psspred_dir, 'temp/seq.dat.ss')
  os.chdir(original_dir)
  return read_ss(path, dataset)

def psipred_ss(dataset):
  psipred_dir = os.environ['PSIPRED_DIR']
  original_dir = os.getcwd()
  os.chdir(psipred_dir)
  system_call('rm -rf ./temp/')
  system_call('mkdir temp')
  os.chdir('./temp')
  pnet.utils.write_dataset(dataset, os.path.join(psipred_dir, 'temp/temp.seq'))
  command = '../runpsipred temp.seq'
  system_call(command)
  path = os.path.join(psipred_dir, 'temp/temp.ss2')
  os.chdir(original_dir)
  return read_ss(path, dataset)
