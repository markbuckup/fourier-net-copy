import os
N_Procs = 10
Pats_per_proc = 150//N_Procs
assert(150 % N_Procs == 0)
for i in range(0,N_Procs):
    os.system('python3 preprocess_radial_faster.py --pat_start {} --pat_end {} --gpu {} &'.format((i*Pats_per_proc)+1, (i+1)*Pats_per_proc, int(i<N_Procs//2)))
