from scipy.io import loadmat
import h5py
import numpy as np
from os import listdir

def test():

	# Locate dataset
	data_files = listdir("/home/ubuntu/data/2013-05-28-4/mat/")

	# Put the spikes and electrode number in here...
	items_list = []
	
	for f in data_files:
#		print("Processing " + str(f))
		electrode_num = None
		if(f.endswith('.mat')):
			temp_inputs = loadmat("/home/ubuntu/data/2013-05-28-4/mat/" + str(f))
			electrode_num = f.split("_")[2]
			input_arr = temp_inputs['temp']
			#print(input_arr.shape)
			#print(electrode_num)
			items_list.append((input_arr, electrode_num))

#	print(items_list[0])
#		print(temp_inputs['temp'])

	labels = loadmat("/home/ubuntu/analysis/2013-05-28-4/data000/data000_celltypes.mat")
	# View some of the data
	print(labels['celltypes'])
	# Get the actual data arrays
	labels_array = labels['celltypes']
	print(labels_array.shape)
	labels_array_reshaped = labels_array.reshape()
	# Now we can modify these arrays to be in the form we need


if __name__=="__main__":
	test()
