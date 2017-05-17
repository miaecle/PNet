#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:38:42 2017

@author: michael
"""

import os
import csv

os.chdir('/home/michael/PNet/datasets')
target_file = './CASP5/casp5.seq'
target_file2 = './CASP5/casp5_seq.csv'
target_path = './CASP5/targets_pdb'
lines = []
with open(target_file, 'r') as f:
    lines = f.readlines()
    
ID = []
Sequences = []
i = 0
current_line = ''
#current_length = 0
for line in lines:
    if line[0] == '>':
        i = i+1
        #assert len(current_line) == current_length
        Sequences.append(current_line)
        current_line = ''
        #current_length = int(line.split()[-2])
        if line[1:6].upper() != line[1:6]:
            print(line[1:6])
        ID.append(line[1:6].upper())
    elif line[0] == '\n':
        pass
    else:
        assert line[-1] == '\n'
        current_line = current_line + line[:-1]

#assert len(current_line) == current_length
Sequences.append(current_line)
Sequences.pop(0)
assert len(ID) == len(Sequences)
Available = [0] * len(ID)

All_available = [a[:5] for a in os.listdir(target_path)]
for i, ID_ in enumerate(ID):
    if ID_ in All_available:
        Available[i] = 1
assert sum(Available) <= len(All_available)

with open(target_file2, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'pdb', 'sequence'])
    for i in range(len(ID)):
        writer.writerow([ID[i], Available[i], Sequences[i]])