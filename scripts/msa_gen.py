import pnet
import os

data = pnet.utils.load_CATH()
print(len(data.IDs))

for file_id in range(40):
    lines = [
        '#!/bin/bash',
        '#SBATCH --job-name=MSA',
        '#SBATCH --time=48:00:00',
        '#SBATCH --partition=jamesz',
        '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=4',
        '#SBATCH --mem-per-cpu=32G',
        'date']
    for i in range(100):
        sample = data.select_by_index([i+file_id*100])
        path = os.path.join(os.environ['PNET_DATA_DIR'], 'MSA', '%s' % sample.IDs[0])
        if not os.path.exists(path):
            os.mkdir(path)
        if os.path.exists('%s/results.fas' % path):
            continue
        pnet.utils.write_dataset(sample, os.path.join(os.environ['PNET_DATA_DIR'], 'MSA', '%s' % sample.IDs[0], 'input.seq'))
        command1 = 'hhblits -v 1 -maxfilt 100000 -realign_max 100000 -all -B 100000 -Z 100000 -diff inf -id 99 -cov 50 -i %s/input.seq -d /scratch/users/zqwu/PNet_data/hhdb/UniRef30_2020_02 -oa3m %s/results.a3m -cpu 4 -n 3 -e 0.001' % (path, path)
        command2 = 'reformat.pl -v 0 -r a3m clu %s/results.a3m %s/results.clu' % (path, path)
        command3 = 'reformat.pl -v 0 -r a3m fas %s/results.a3m %s/results.fas' % (path, path)
        lines.append(command1)
        lines.append(command2)
        lines.append(command3)
        lines.append('date')
    with open('job%d.sh' % file_id, 'w') as f:
        for l in lines:
            f.write(l + '\n')
