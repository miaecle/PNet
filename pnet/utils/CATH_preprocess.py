import os
import numpy as np
import pickle
import csv

###############################################################################

cath_entries = {}
with open(os.path.join(os.environ['PNET_DATA_DIR'],
                       'CATH',
                       'cath-domain-list.txt'), 'r') as f:
  for line in f:
    if line.startswith('#'):
      continue
    line = line.split()
    cath_entries[line[0]] = (list(map(int, line[1:-1])), float(line[-1]))

with open(os.path.join(os.environ['PNET_DATA_DIR'],
                       'CATH',
                       'cath-domain-list.pkl'), 'wb') as f:
  pickle.dump(cath_entries, f)


cath_seqs = {}
entry = {}
in_seg = False
with open(os.path.join(os.environ['PNET_DATA_DIR'],
                       'CATH',
                       'cath-domain-description-file.txt'), 'r') as f:
  
  for line in f:
    if line.startswith('#'):
      continue
    if line.startswith('//'):
      assert at_seg == n_seg
      entry["SEGS"] = segments
      in_seg = False
      cath_seqs[entry['DOMAIN']] = entry
      entry = {}
      continue
    if line.startswith("ENDSEG"):
      at_seg += 1
      continue
    
    split_ind = line.find(' ')
    k = line[:split_ind].strip()
    v = line[split_ind:].strip()
    
    if k == 'NSEGMENTS':
      n_seg = int(v)
      at_seg = 0
      in_seg = True
      segments = [{} for _ in range(n_seg)]
      continue
    
    if not in_seg:
      if not k in entry:
        entry[k] = v
      else:
        entry[k] = entry[k] + v
    else:
      if not k in segments[at_seg]:
        segments[at_seg][k] = v
      else:
        segments[at_seg][k] = segments[at_seg][k] + v

with open(os.path.join(os.environ['PNET_DATA_DIR'],
                       'CATH',
                       'cath-domain-description-file.pkl'), 'wb') as f:
  pickle.dump(cath_seqs, f)

###############################################################################

cath_entries = pickle.load(open(os.path.join(os.environ['PNET_DATA_DIR'],
                                'CATH',
                                'cath-domain-list.pkl'), 'rb'))

cath_seqs = pickle.load(open(os.path.join(os.environ['PNET_DATA_DIR'],
                             'CATH',
                             'cath-domain-description-file.pkl'), 'rb'))

# LEVEL 1: 4
# LEVEL 2: 41
# LEVEL 3: 1391
# LEVEL 4: 6119
# LEVEL 5: 31289
# LEVEL 6: 43287
# LEVEL 7: 58178
# LEVEL 8: 111976
# LEVEL 9: 434857

af_train = {}
with open(os.path.join(os.environ['PNET_DATA_DIR'],
                                'CATH',
                                'train_domains.txt', 'r') as f:
  for line in f:
    k = line.strip()
    if k in cath_entries:
      code = tuple(cath_entries[k][0][:5])
      if code in af_train:
        af_train[code].append(k)
      else:
        af_train[code] = [k]

af_test = {}
with open(os.path.join(os.environ['PNET_DATA_DIR'],
                                'CATH',
                                'test_domains.txt', 'r') as f:
  for line in f:
    k = line.strip()
    if k in cath_entries:
      code = tuple(cath_entries[k][0][:5])
      if code in af_test:
        af_test[code].append(k)
      else:
        af_test[code] = [k]

mapping = {}
for k in cath_entries:
  code = tuple(cath_entries[k][0][:5])
  if not code in mapping:
    mapping[code] = [k]
  else:
    mapping[code].append(k)

np.random.seed(123)
assembled_train_samples = []
assembled_test_samples = []
for k in mapping:
  if k in af_train:
    sample = np.random.choice(af_train[k])
    ct = 0
    while len(cath_seqs[sample]['DSEQS']) != int(cath_seqs[sample]['DLENGTH']) and ct < 10:
      sample = np.random.choice(mapping[k])  
      ct += 1
    if len(cath_seqs[sample]['DSEQS']) == int(cath_seqs[sample]['DLENGTH']):
      assembled_train_samples.append((sample, cath_seqs[sample]['DSEQS']))
  elif k in af_test:
    sample = np.random.choice(af_test[k])
    ct = 0
    while len(cath_seqs[sample]['DSEQS']) != int(cath_seqs[sample]['DLENGTH']) and ct < 10:
      sample = np.random.choice(mapping[k])  
      ct += 1
    if len(cath_seqs[sample]['DSEQS']) == int(cath_seqs[sample]['DLENGTH']):
      assembled_test_samples.append((sample, cath_seqs[sample]['DSEQS']))
  else:
    sample = np.random.choice(mapping[k])
    ct = 0
    while len(cath_seqs[sample]['DSEQS']) != int(cath_seqs[sample]['DLENGTH']) and ct < 10:
      sample = np.random.choice(mapping[k])  
      ct += 1
    if len(cath_seqs[sample]['DSEQS']) == int(cath_seqs[sample]['DLENGTH']):
      assembled_train_samples.append((sample, cath_seqs[sample]['DSEQS']))

assembled_train_samples = sorted(assembled_train_samples, key=lambda x: x[0])
assembled_test_samples = sorted(assembled_test_samples, key=lambda x: x[0])

with open(os.path.join(os.environ['PNET_DATA_DIR'], 'CATH_selected.csv'), 'w') as f:
  writer = csv.writer(f)
  writer.writerow(['DomainID', 'Sequence', 'Split'])
  for pair in assembled_train_samples:
    writer.writerow([pair[0], pair[1], 0])
  for pair in assembled_test_samples:
    writer.writerow([pair[0], pair[1], 1])

