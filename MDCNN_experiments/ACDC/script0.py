import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', nargs='+', type = int, default = [-1])
args = parser.parse_args()

GPUSTR = ' '.join([str(x) for x in args.gpu])

task_list = ['1.0.0.1', '1.0.0.2', '1.0.0.3', '1.0.0.4', '1.0.0.5', '1.0.0.6', '1.0.1.1', '1.0.1.2', '1.0.1.3', '1.0.1.4', '1.0.1.5', '1.0.1.6', '1.0.1.7', '1.0.1.8', '1.0.2.1', '1.0.2.2', '1.0.2.3', '1.0.2.4', '1.0.2.5', '1.0.2.6', '1.0.2.7']

prev_task = None
for task in task_list:
	if not prev_task is None:
		if not os.path.isfile(os.path.join(prev_task, 'status.txt')):
			print("\n\n{} did not complete successfully".format(prev_task))
			print("Breaking Operation", flush = True)
			break
	os.system("rm status.txt 2> /dev/null")
	print("Now Running Task {}".format(task), flush = True)
	# os.system('python3 -Wignore main.py --port 12355 --run_id {} --neptune_log --gpu {} > {}/train.log 2>&1 && python3 -Wignore main.py --port 12355 --run_id {} --neptune_log --gpu {} --resume --eval > {}/test.log 2>&1'.format(task, GPUSTR, task, task, GPUSTR, task))
	os.system('python3 -Wignore main.py --port 12355 --run_id {} --neptune_log --gpu {} --resume --eval --visualise_only > {}/test.log 2>&1'.format(task, GPUSTR, task, task, GPUSTR, task))
	print("Task {} Terminated".format(task), flush = True)
	prev_task = task
	