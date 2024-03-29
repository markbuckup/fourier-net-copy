import os
for i in range(131,141):
    os.system('python3 preprocess_radial_faster.py --pat_start {} --pat_end {} --gpu 1 &'.format(i, i+1))
