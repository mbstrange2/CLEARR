from naive_multiwire_tensor import *

# Author: Dante Muratore


# Paths:
minute = 0
data_path = '/drives/c/Users/Mek/DATA/2012_1/'
#analysis_path = '/home/ubuntu/analysis/'
#dataset = '2013-05-28-4/'
inp_file = data_path + f'data000_300_sample_start_{minute}m.mat'

strategy = 'naive_2ramp/'
#subset = 'data000'
#vision_path = '/home/ubuntu/vision7-unix'
#vision_rgb = vision_path + '/RGB-8-1-0.48-11111.xml'

# Inputs:
time_window = 300 # number of time samples in data in seconds
chunk_size = 1000 # chunk of samples to be processed simultaneously
num_bits = [8]
n_wires = [1]
ovr = True
#save_path = data_path + dataset + strategy + str(num_bits) + 'b_' + str(n_wires) + 'w/' + subset

for b in range(len(num_bits)):
    for w in range(len(n_wires)):
        print('Running naive decoder with ' + str(num_bits[b]) + ' bits and ' + str(n_wires[w]) + " wires")
        #save_path = data_path + dataset + strategy + str(num_bits[b]) + 'b_' + str(n_wires[w]) + 'w/' + subset
        save_path = data_path + f'min_{minute}/'
        naive_multiwire_tensor(inp_file, save_path, time_window, chunk_size, num_bits[b], n_wires[w], ovr)
        #naive_interleaved_tensor(inp_file, save_path, time_window, chunk_size, num_bits[b], n_wires[w], ovr)
