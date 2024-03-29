import os
for i in range(1,6):
    os.system('python3 preprocess_radial_faster.py --pat_start {} --pat_end {} --gpu 1 & > {}.log'.format(i, i+1, i))
