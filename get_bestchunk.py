import sys

if len(sys.argv) != 2:
    print('Provide input file (output of get_timechunks.py).', file=sys.stderr)
    sys.exit(1)

validpercent_max = 0
remove_list_max = None
params_max = ''

with open(sys.argv[1]) as infile:

    for line in infile:
        if line.startswith('Removing'):
            remove_list = eval(line[9:])
        else:
            parts = line.rsplit(',', 1)
            percent = float(parts[1].strip()[:-1])
            if percent > validpercent_max:
                validpercent_max = percent
                remove_list_max = remove_list
                params_max = line.strip()

print('Max valid percent:', validpercent_max)
print('Remove list:', remove_list_max)
print('Details:', params_max)
