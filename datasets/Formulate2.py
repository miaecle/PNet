#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:12:09 2017

@author: michael
"""

import os
import numpy as np
os.chdir('/home/michael/PNet/datasets')
target_file = './CASP5/casp5.seq'
sequence_dir = './CASP5/sequences'
all_sequences = os.listdir(sequence_dir)
all_sequences.sort()
raw_lines=[]
for file_name in all_sequences:
    with open(sequence_dir+'/'+file_name, 'r') as f:
        raw_lines.extend(f.readlines())

all_lines = []
current_sequence = ''
for line in raw_lines:
    if line[0] == '>':
        all_lines.append(current_sequence+'\n')
        current_sequence = ''
        all_lines.append(line)
    else:
        line = line.replace(" ", "")
        current_sequence = current_sequence + line[:-1]
all_lines.append(current_sequence+'\n')
all_lines.pop(0)

out_lines = []
for line_id, line in enumerate(all_lines):
    if line[0] == '>':
        if len(out_lines) > 2:
            if out_lines[-2] != '\n' and out_lines[-1] == '\n':
                out_lines.append('\n')
            elif out_lines[-1] != '\n':
                out_lines.append('\n')
                out_lines.append('\n')
        out_lines.append(line)
    elif line[0] == '\n':
        pass
    else:
        if len(line) <= 61 and line[-1] == '\n':
            out_lines.append(line)
        else:
            n_lines = int(np.ceil(float(len(line)-1) / 60))
            for i in range(n_lines-1):
                out_lines.append(line[i*60:(i+1)*60] + '\n')
            out_lines.append(line[(n_lines-1)*60:])
a = [len(i) for i in out_lines]

with open(target_file, 'w') as f:
    f.writelines(out_lines)