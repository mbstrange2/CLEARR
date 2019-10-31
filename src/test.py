from scipy.io import loadmat
import h5py
import numpy as np
from os import listdir

import os, sys, pdb, pickle
from multiprocessing import Pool
import numpy as np
#import samplerate
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt

def test():

	labels = loadmat("/home/ubuntu/analysis/2013-05-28-4/data000/data000_celltypes.mat")
	# Get the actual data arrays
	labels_array = labels['celltypes']
#	print(labels_array.shape)
	labels_array_reshaped = labels_array.reshape((557, 1))
	labels_array = labels_array_reshaped[0:512]
#	print(labels_array)
	label_np = np.array(labels_array)
	# Locate dataset
	data_files = listdir("/home/ubuntu/data/2013-05-28-4/mat/")

	# Put the spikes and electrode number in here...
	items_list = []
	
	for f in data_files:
#		print("Processing " + str(f))
		electrode_num = None
		if(f.endswith('.mat')):
			temp_inputs = loadmat("/home/ubuntu/data/2013-05-28-4/mat/" + str(f))
			electrode_num = int(f.split("_")[2])
			input_arr = temp_inputs['temp']
			#print(input_arr.shape)
			#print(electrode_num)
			items_list.append((input_arr.squeeze(), label_np[electrode_num-1][0][0]))

	items_np = np.array(items_list)
	print(items_np[0])
#	print(items_np[0].shape)

	# Now we can modify these arrays to be in the form we need
	plt.figure(figsize=(20, 18))
	for i in range(6):
	#	(rate, data) = wav.read(train_data_dir + fnames[i])
		plt.subplot(3,2,i+1)
		spec = plt.specgram(items_np[i][0].squeeze(), Fs=20000, NFFT=5, noverlap=1, mode='psd')
		plt.yscale('log')
		plt.ylim([10,8000])
		plt.title(data_files[i])
	plt.show()

if __name__=="__main__":
	test()
