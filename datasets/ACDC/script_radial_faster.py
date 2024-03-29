for i in range(15):
    os.system('python3 preprocess_radial_faster.py --pat_start {} --pat_end {}'.format(i*10 + 1, i*10+11))
