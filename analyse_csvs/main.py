import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

mdcnn_ssims = []
mdcnn_mse = []
with open('mdcnn.csv', 'r') as f:
	for i,line in enumerate(f):
		if i == 0:
			continue
		mdcnn_ssims.append(float(line.strip().split(',')[3]))
		mdcnn_mse.append(float(line.strip().split(',')[5]))

lstm_ssims = []
lstm_mse = []
with open('lstm.csv', 'r') as f:
	for i,line in enumerate(f):
		if i == 0:
			continue
		lstm_ssims.append(float(line.strip().split(',')[3]))
		lstm_mse.append(float(line.strip().split(',')[5]))
mdcnn_ssims = np.array(mdcnn_ssims)
lstm_ssims = np.array(lstm_ssims)
mdcnn_mse = np.array(mdcnn_mse)**0.5
lstm_mse = np.array(lstm_mse)**0.5
mdcnn_spf = np.array(scipy.io.loadmat('mdcnn.mat')['times'][0][1:])
lstm_spf = np.array(scipy.io.loadmat('lstm.mat')['times'][0][1:])

print('MDCNN Average MSE = {} +- {}'.format(mdcnn_mse.mean(), mdcnn_mse.std()))
print('LSTM Average MSE = {} +- {}'.format(lstm_mse.mean(), lstm_mse.std()))

print('MDCNN SSIM Mean = {} +- {}'.format(mdcnn_ssims.mean(), mdcnn_ssims.std()))
print('LSTM SSIM Mean = {} +- {}'.format(lstm_ssims.mean(), lstm_ssims.std()))

print('MDCNN SSIM median = {}'.format(np.median(mdcnn_ssims)))
print('LSTM SSIM median = {}'.format(np.median(lstm_ssims)))

print('MDCNN SSIM mode = {}'.format(scipy.stats.mode(mdcnn_ssims, keepdims = True)[0][0]))
print('LSTM SSIM mode = {}'.format(scipy.stats.mode(lstm_ssims, keepdims = True)[0][0]))


plt.figure()
plt.style.use('seaborn-deep')
plt.hist([mdcnn_ssims, lstm_ssims], bins=np.linspace(0.65, 1, 30), label = ['MD-CNN', 'convLSTM'])
plt.xlabel('SSIM')
plt.ylabel('n samples')
plt.legend()
plt.savefig('hist_SSIM.jpg')
plt.figure()
plt.style.use('seaborn-deep')
plt.hist([1/mdcnn_spf, 1/lstm_spf], bins=30, label = ['MD-CNN', 'convLSTM'])
plt.xlabel('FPS')
plt.ylabel('n samples')
plt.legend()
plt.savefig('hist_FPS.jpg')

font = {'family' : 'normal',
        'size'   : 14}
import matplotlib
matplotlib.rc('font', **font)

plt.figure()
plt.boxplot([1/mdcnn_spf, 1/lstm_spf], sym = '+', whis = [0,100])
plt.xticks([1, 2], ['MD-CNN', 'convLSTM'])
plt.ylabel('FPS')
ax = plt.gca()
plt.savefig('boxplots_FPS.jpg')
plt.figure()
plt.boxplot([mdcnn_ssims, lstm_ssims], sym = '+', whis = [0,100])
plt.xticks([1, 2], ['MD-CNN', 'convLSTM'])
plt.ylabel('SSIM')
ax = plt.gca()
ax.set_ylim([0,1])
plt.savefig('boxplots_SSIM.jpg')

print('MDCNN time per frame = {} +- {}'.format(mdcnn_spf.mean(), mdcnn_spf.std()))
print('LSTM time per frame = {} +- {}'.format(lstm_spf.mean(), lstm_spf.std()))

print('MDCNN FPS = {} +- {}'.format((1/mdcnn_spf).mean(), (1/mdcnn_spf).std()))
print('LSTM FPS = {} +- {}'.format((1/lstm_spf).mean(), (1/lstm_spf).std()))