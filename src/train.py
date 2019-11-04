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
from tensorflow import keras
from tensorflow.keras import layers

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
    
    n_x = len(raw_arr[:,0])
    #print(n_x)
    n_y = n_x

    raw_arr_norm = tf.keras.utils.normalize(raw_arr, axis=0)
    comp_arr_norm = comp_arr

    raw_arr_norm = raw_arr_norm.T
    comp_arr_norm = comp_arr_norm.T

    raw_train = tf.dtypes.cast(raw_arr_norm[0:12000], tf.float32)
    raw_dev = tf.dtypes.cast(raw_arr_norm[12000:], tf.float32)
    comp_train = tf.dtypes.cast(comp_arr_norm[0:12000], tf.float32)
    comp_dev = tf.dtypes.cast(comp_arr_norm[12000:], tf.float32)

    def model_create(example_length):
        inputs = keras.Input(shape=(example_length,), name="in")
        x = layers.Dense(1000, activation="relu", name="dense_1")(inputs)
        x = layers.Dense(1000, activation="relu", name="dense_2")(x)
        x = layers.Dense(1000, activation="relu", name="dense_3")(x)
        x = layers.Dense(1000, activation="relu", name="dense_4")(x)
        x = layers.Dense(1000, activation="relu", name="dense_5")(x)
        outputs = layers.Dense(example_length, activation="tanh", name="out")(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    model = model_create(n_x)

    use_chk = False

    checkpoint_filepath = "./model_chk.hdf5"
    training = False

    if os.path.exists(checkpoint_filepath) and use_chk is True:
        print("Reading checkpointed model...")
        model.load_weights(checkpoint_filepath)
    else:
        print("Training model...")
        training = True

#    model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
    model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.MeanSquaredError(),
        # List of metrics to monitor
        metrics=[keras.metrics.MeanSquaredError()])

    if training is True:
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        history = model.fit(comp_train, raw_train,
                    batch_size=128,
                    epochs=5,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(comp_dev, raw_dev),
                    callbacks=callbacks_list)



    plot = True

    if plot is True:
        comp_5 = comp_train[5]
        comp5_re = np.reshape(comp_train[5], (1, 71))
        predict_5 = model.predict(comp5_re)
        predict_5 = predict_5.reshape((71,1))
        print(predict_5)
        actual_5 = raw_train[5]
        plt.figure(1)
        plt.title('Compressed...')
        plt.plot(comp_5)
        plt.figure(2)
        plt.title('Recon...')
        plt.plot(predict_5)
        plt.figure(3)
        plt.title('Raw...')
        plt.plot(actual_5)
        plt.show()

if __name__=="__main__":
    train()

    # @tf.function
    # def forward(x):
    #     Z1 = tf.matmul(W1, x) + b1
    #     A1 = tf.nn.relu(Z1)
    #     Z2 = tf.matmul(W2, A1) + b2
    #     A2 = tf.nn.relu(Z2)
    #     Z3 = tf.matmul(W3, A2) + b3
    #     A3 = tf.tanh(Z3)

    #     return A3

    # @tf.function
    # def cost(forward, Y):
    #     return tf.reduce_mean((forward-Y)**2)


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