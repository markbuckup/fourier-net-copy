import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type = int, default = '0')
args = parser.parse_args()

task_list = ['1.0']

prev_task = None
for task in task_list:
	if not prev_task is None:
		if not os.path.isfile(os.path.join(prev_task, 'status.txt')):
			print("\n\n{} did not complete successfully".format(prev_task))
			print("Breaking Operation", flush = True)
			break
	os.chdir(task)
	os.system("rm status.txt 2> /dev/null")
	print("Now Running Task {}".format(task), flush = True)
	os.system('python3 -Wignore main.py --gpu {} > train.log 2>&1 && python3 -Wignore main.py --gpu {} --resume --eval > test.log 2>&1'.format(args.gpu, args.gpu))
	# os.system('python3 -Wignore main.py --gpu {} --resume --eval > test.log 2>&1'.format(args.gpu))
	print("Task {} Terminated".format(task), flush = True)
	prev_task = task
	os.chdir('../')
