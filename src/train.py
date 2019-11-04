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
import tensorflow as tf

def reformat_data(raw_path, comp_path, ignore_list, raw_array_path, comp_array_path):
    raw_files = listdir(raw_path)
    comp_files = listdir(comp_path)
    # Put the spikes and electrode number in here...
    raw_dict = {}
    comp_dict = {}
    
    for f in raw_files:
        electrode_num = None
        if f.endswith('.mat') and f.split("_")[3] == "spike":
            f = f.split(".")[0] # Get rid of file extension
            #print("Processing " + str(f))
            temp_inputs = loadmat(raw_path + str(f))
            electrode_num = int(f.split("_")[2])
            spike_num = int(f.split("_")[4])
            input_arr = temp_inputs['temp']
            if electrode_num not in ignore_list:
                raw_dict[(electrode_num, spike_num)] = input_arr.squeeze()

    for f in comp_files:
        electrode_num = None
        if f.endswith('.mat') and f.split("_")[3] == "spike":
            f = f.split(".")[0] # Get rid of file extension
            #print("Processing " + str(f))
            temp_inputs = loadmat(comp_path + str(f))
            electrode_num = int(f.split("_")[2])
            spike_num = int(f.split("_")[4])
            input_arr = temp_inputs['temp']
            if electrode_num not in ignore_list:
                comp_dict[(electrode_num, spike_num)] = input_arr.squeeze()

    num_examples = len(raw_dict.keys())
    length_example = len(raw_dict[(1,1)])

    raw_arr = np.zeros((length_example, num_examples))
    comp_arr = np.zeros((length_example, num_examples))

    i = 0
    for (elec_num, spike_num) in raw_dict.keys():
        #print("Electrode: " + str(elec_num) + "\tSpike: " + str(spike_num))
        if (elec_num, spike_num) not in comp_dict.keys():
            print("FAILURE...not in comp_dict")
            break
        raw_arr[:,i] = raw_dict[(elec_num, spike_num)]
        comp_arr[:,i] = comp_dict[(elec_num, spike_num)]
        i += 1

    np.save(raw_array_path, raw_arr)
    np.save(comp_array_path, comp_arr)

def train():

    raw_path = "/Users/max/Documents/CS230/data/raw/" # Change this for yourself
    comp_path = "/Users/max/Documents/CS230/data/compressed/" # Change this for yourself
    ignore_list = [45, 49, 177, 401, 418, 460]

    raw_array_path = "./raw_array.npy"
    comp_array_path = "./comp_array.npy"

    if not os.path.exists(raw_array_path) or not os.path.exists(comp_array_path):
        print("np files don't exist, creating now...")
        reformat_data(raw_path, comp_path, ignore_list, raw_array_path, comp_array_path)
    else:
        print("NP array files already exist, proceeding as usual...")

    raw_arr = np.load(raw_array_path)
    comp_arr = np.load(comp_array_path)
    print("Raw dimensions")
    print(raw_arr.shape)
    print(raw_arr[0])
    print(raw_arr[0].shape)

    # Now we can modify these arrays to be in the form we need
    # plt.figure(figsize=(20, 18))
    # for i in range(6):
    # #	(rate, data) = wav.read(train_data_dir + fnames[i])
    #     plt.subplot(3,2,i+1)
    #     spec = plt.specgram(items_np[i][0].squeeze(), Fs=20000, NFFT=5, noverlap=1, mode='psd')
    #     plt.yscale('log')
    #     plt.ylim([10,8000])
    #     plt.title(raw_list[i])
    # plt.show()

if __name__=="__main__":
    train()