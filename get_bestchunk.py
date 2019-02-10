import glob

flist = glob.iglob('validcounts_cutoff*_*M.txt')

validpercent_list = []
remove_list_list = []
params_list = []
cutoff_list = []

for fpath in flist:
    cutoff = int(fpath.split('_')[1][-2:])
    with open(fpath) as infile:
        for line in infile:
            if line.startswith('Removing'):
                remove_list = eval(line[9:])
            else:
                parts = line.rsplit(',', 1)
                percent = float(parts[1].strip()[:-1])
                if percent >= 50:
                    validpercent_list.append(percent)
                    remove_list_list.append(remove_list)
                    params_list.append(parts[0].strip()[5:])
                    cutoff_list.append(cutoff)

with open('validcounts_bestchunks.txt', 'w') as fout:
    for tup in zip(validpercent_list, remove_list_list, params_list, cutoff_list):
        validpercent, remove_list, params, cutoff = tup
        fout.write('\nMax valid percent: {}\n'.format(validpercent))
        fout.write('Remove count: {}\n'.format(len(remove_list)))
        fout.write('Remove list: {}\n'.format(remove_list))
        fout.write('Removal cutoff percent: {}\n'.format(cutoff))
        fout.write('Details: {}\n'.format(params))
