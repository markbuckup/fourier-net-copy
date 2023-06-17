import os

otfile = open('results.csv', 'w')
otfile.write('Run ID, Recon Loss, Recon-FT Loss, FT Loss, Train SSIM, Test SSIM, Train L1, Test L1, Train RMSE, Test RMSE\n')

for run_ids in sorted(os.listdir('.')):
	if (not os.path.isdir(run_ids)) or run_ids.endswith('faulty') or run_ids.startswith('.'):
		continue
	mydic = {}
	if not os.path.isfile(os.path.join(run_ids, 'test.log')):
		continue
	with open(os.path.join(run_ids, 'params.py'), 'r') as f:
		for line in f:
			line = line.strip()
			if line.startswith('#') or 'batch_size' in line:
				continue
			elif '=' in line and ']' in line:
				lhs = line.strip().split('=')[0].strip().split("'")[1].strip()
				rhs = line.strip().split('=')[1].strip().replace("'", "")
				mydic[lhs] = rhs
	ssims = []
	l1s = []
	l2s = []
	with open(os.path.join(run_ids, 'test.log'), 'r') as f:
		for line in f:
			line = line.strip()
			if line.startswith('#') or 'batch_size' in line:
				continue
			if "SSIM Score =" in line:
				ssims.append(float(line.split('=')[1].strip()))
			if "L1 Loss =" in line:
				l1s.append(float(line.split('=')[1].strip()))
			if "L2 Loss =" in line:
				l2s.append(float(line.split('=')[1].strip()))
		otfile.write(run_ids)
		otfile.write(',')
		otfile.write(mydic['loss_recon'])
		otfile.write(',')
		otfile.write(mydic['loss_reconstructed_FT'])
		otfile.write(',')
		otfile.write(mydic['loss_FT'])
		otfile.write(',')
		otfile.write(str(ssims[1]))
		otfile.write(',')
		otfile.write(str(ssims[0]))
		otfile.write(',')
		otfile.write(str(l1s[1]))
		otfile.write(',')
		otfile.write(str(l1s[0]))
		otfile.write(',')
		otfile.write(str(l2s[1]**0.5))
		otfile.write(',')
		otfile.write(str(l2s[0]**0.5))
		otfile.write('\n')
		otfile.flush()

otfile.flush()
otfile.close()