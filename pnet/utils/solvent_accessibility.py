#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 23:18:58 2017

@author: zqwu
"""

import numpy as np
import os
import pandas as pd
import pnet

from pnet.utils import system_call

def string_to_onehot_sa(sa):
  state = {'B': [1, 0, 0], 'M': [0, 1, 0], 'E': [0, 0, 1]}
  return np.array([state[a] for a in sa])

def read_sa(path, dataset):
  with open(path, 'r') as f:
    lines = f.readlines()
  lines = [line.split() for line in lines]
  start = 0
  length = len(dataset.sequences[0])
  for i, line in enumerate(lines):
    if len(line) == 6 and line[0] == '1' and line[2] in ['B', 'M', 'E']:
      start = i
      break
  data = np.array(lines[start:start+length])
  seq = ''.join(list(data[:,1]))
  assert str(seq) == str(dataset.sequences[0])
  bury_defined = 1
  medium_defined = 1
  exposed_defined = 1
  current_line = 0
  order = np.zeros(3)
  while bury_defined + medium_defined + exposed_defined > 1:
    if data[current_line, 2] == 'B' and bury_defined > 0:
      values = np.array(data[current_line, 3:6])
      a = np.argmax(values)
      if values[a-1] < values[a] and values[a-2] < values[a]:
        order[0] = a + 3
        bury_defined = 0
    elif data[current_line, 2] == 'M' and medium_defined > 0:
      values = np.array(data[current_line, 3:6])
      a = np.argmax(values)
      if values[a-1] < values[a] and values[a-2] < values[a]:
        order[1] = a + 3
        medium_defined = 0
    elif data[current_line, 2] == 'E' and exposed_defined > 0:
      values = np.array(data[current_line, 3:6])
      a = np.argmax(values)
      if values[a-1] < values[a] and values[a-2] < values[a]:
        order[2] = a + 3
        exposed_defined = 0
    if bury_defined + medium_defined + exposed_defined == 1:
      order[np.argmin(order)] = 12 - np.sum(order)
    current_line = current_line + 1
  assert sorted(order) == [3, 4, 5]
  order = np.array(order, dtype=int)
  return np.array(np.stack([data[:, order[0]],
                            data[:, order[1]],
                            data[:, order[2]]], axis=1), dtype=float)

def raptorx_sa(dataset):
  raptorx_dir = os.environ['RAPTORX_DIR']
  original_dir = os.getcwd()
  os.chdir(raptorx_dir)
  system_call('rm temp.seq')
  system_call('rm -rf ./tmp/temp/')
  pnet.utils.write_dataset(dataset, os.path.join(raptorx_dir, 'temp.seq'))
  command = './oneline_command.sh temp.seq'
  system_call(command)
  path = os.path.join(raptorx_dir, 'tmp/temp/temp.acc')
  os.chdir(original_dir)
  return read_sa(path, dataset)

def netsurfp_rsa(dataset):
  netsurfp_dir = os.environ['NETSURFP_DIR']
  original_dir = os.getcwd()
  os.chdir(netsurfp_dir)
  system_call('rm -rf ./temp/')
  system_call('mkdir temp')
  os.chdir('./temp')
  pnet.utils.write_dataset(dataset, os.path.join(netsurfp_dir, 'temp/temp.seq'))
  command = '../netsurfp -i temp.seq -a >temp.rsa'
  system_call(command)
  path = os.path.join(netsurfp_dir, 'temp/temp.rsa')
  os.chdir(original_dir)
  with open(path, 'r') as f:
    lines = f.readlines()
  lines = [line.split() for line in lines]
  start = 0
  length = len(dataset.sequences[0])
  for i, line in enumerate(lines):
    if len(line) == 10 and line[3] == '1' and line[0] in ['B', 'E']:
      start = i
      break
  data = np.array(lines[start:start+length])
  seq = ''.join(list(data[:,1]))
  assert str(seq) == str(dataset.sequences[0])
  rsa = data[:, 4]
  return np.array(np.reshape(rsa, newshape=(-1,1)), dtype=float)
