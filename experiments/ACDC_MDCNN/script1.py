import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', nargs='+', type = int, default = [-1])
args = parser.parse_args()

GPUSTR = ' '.join([str(x) for x in args.gpu])

task_list = ['1.1.0.4','1.1.0.5','1.1.0.6']

prev_task = None
for task in task_list:
	if not prev_task is None:
		if not os.path.isfile(os.path.join(prev_task, 'status.txt')):
			print("\n\n{} did not complete successfully".format(prev_task))
			print("Breaking Operation", flush = True)
			break
	os.system("rm status.txt 2> /dev/null")
	print("Now Running Task {}".format(task), flush = True)
	# os.system('python3 -Wignore main.py --port 12356 --run_id {} --neptune_log --gpu {} > {}/train.log 2>&1 && python3 -Wignore main.py --port 12356 --run_id {} --neptune_log --gpu {} --resume --eval > {}/test.log 2>&1'.format(task, GPUSTR, task, task, GPUSTR, task))
	os.system('python3 -Wignore main.py --port 12356 --run_id {} --gpu {} > {}/train.log 2>&1 && python3 -Wignore main.py --port 12356 --run_id {} --gpu {} --resume --eval > {}/test.log 2>&1'.format(task, GPUSTR, task, task, GPUSTR, task))
	print("Task {} Terminated".format(task), flush = True)
	prev_task = task
	