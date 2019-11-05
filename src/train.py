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
        #if f.endswith('.mat') and f.split("_")[3] == "spike":
        if f.endswith('.mat'):
            f = f.split(".")[0] # Get rid of file extension
            #print("Processing " + str(f))
            temp_inputs = loadmat(raw_path + str(f))
            electrode_num = int(f.split("_")[2])
            spike_num = int(f.split("_")[4])
            spikedness = f.split("_")[3]
            input_arr = temp_inputs['temp']
            if electrode_num not in ignore_list:
                raw_dict[(electrode_num, spike_num, spikedness)] = input_arr.squeeze()

    for f in comp_files:
        electrode_num = None
      #  if f.endswith('.mat') and f.split("_")[3] == "spike":
        if f.endswith('.mat'):
            f = f.split(".")[0] # Get rid of file extension
            #print("Processing " + str(f))
            temp_inputs = loadmat(comp_path + str(f))
            electrode_num = int(f.split("_")[2])
            spike_num = int(f.split("_")[4])
            spikedness = f.split("_")[3]
            input_arr = temp_inputs['temp']
            if electrode_num not in ignore_list:
                comp_dict[(electrode_num, spike_num, spikedness)] = input_arr.squeeze()


    num_examples = len(raw_dict.keys())

    print(num_examples)

    length_example = len(raw_dict[(1,1, "nonspike")])
    length_example = len(raw_dict[(1,1, "spike")])

    raw_arr = np.zeros((length_example, num_examples))
    comp_arr = np.zeros((length_example, num_examples))

    i = 0
    for (elec_num, spike_num, spikedness) in raw_dict.keys():
        #print("Electrode: " + str(elec_num) + "\tSpike: " + str(spike_num))
        if (elec_num, spike_num, spikedness) not in comp_dict.keys():
            print("FAILURE...not in comp_dict")
            break
        raw_arr[:,i] = raw_dict[(elec_num, spike_num)]
        comp_arr[:,i] = comp_dict[(elec_num, spike_num)]
        i += 1

    np.save(raw_array_path, raw_arr)
    np.save(comp_array_path, comp_arr)


def train(use_chk=False, plot=False, show=False, save=False, plot_idx=0):
    raw_path = "/Users/max/Documents/CS230/data/raw/" # Change this for yourself
    comp_path = "/Users/max/Documents/CS230/data/compressed/" # Change this for yourself
    ignore_list = [45, 49, 177, 401, 418, 460]

    raw_array_path = "./raw_array.npy"
    comp_array_path = "./comp_array.npy"

    remake = True

    if not os.path.exists(raw_array_path) or not os.path.exists(comp_array_path) or remake is True:
        print("np files don't exist, creating now...")
        reformat_data(raw_path, comp_path, ignore_list, raw_array_path, comp_array_path)
    else:
        print("NP array files already exist, proceeding as usual...")

    raw_arr = np.load(raw_array_path)
    comp_arr = np.load(comp_array_path)
    
    n_x = len(raw_arr[:,0])
    n_y = n_x

   # raw_arr_norm = tf.keras.utils.normalize(raw_arr, axis=1)
    raw_arr_norm = raw_arr
    comp_arr_norm = comp_arr / 255
    #comp_arr_norm = comp_arr

    raw_arr_norm = raw_arr_norm.T
    comp_arr_norm = comp_arr_norm.T

    # raw_arr_norm = raw_arr_norm[:, :, None]
    # #print(raw_arr_norm.shape)
    # comp_arr_norm = comp_arr_norm[ :, :, None ]

    raw_arr_norm_backup = raw_arr_norm
    comp_arr_norm_backup = comp_arr_norm

    raw_arr_norm = raw_arr_norm[:, None, :]
    comp_arr_norm = comp_arr_norm[:, None, :]

    raw_train = tf.dtypes.cast(raw_arr_norm[0:12000], tf.float32)
    raw_dev = tf.dtypes.cast(raw_arr_norm[12000:], tf.float32)
    comp_train = tf.dtypes.cast(comp_arr_norm[0:12000], tf.float32)
    comp_dev = tf.dtypes.cast(comp_arr_norm[12000:], tf.float32)

    def model_create(example_length):
        inputs = keras.Input(shape=(1, example_length, ), name="in")
        x = layers.Conv1D(filters=10, kernel_size=5, strides=1, padding='same', activation=None,data_format='channels_first', use_bias=True, kernel_initializer='glorot_uniform')(inputs)
      #  x = layers.Dense(1000, activation="relu", name="dense_1")(inputs)
      #  x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.Dense(1500, activation="relu", name="dense_2")(x)
        x = layers.Dense(1500, activation="relu", name="dense_3")(x)
        x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.Dense(1500, activation="relu", name="dense_4")(x)
        x = layers.Dense(1500, activation="relu", name="dense_5")(x)
        #x = layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation=None,data_format='channels_first', use_bias=True, kernel_initializer='glorot_uniform')(inputs)
        x = layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation=None, data_format='channels_first', kernel_initializer='glorot_uniform')(x)
        #outputs = layers.Dense(example_length, activation="tanh", name="out")(x)
        outputs = layers.Dense(example_length, name="out")(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    model = model_create(n_x)

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
        #loss=keras.losses.MeanAbsoluteError(),
        loss=keras.losses.MeanSquaredError(),
        # List of metrics to monitor
        #metrics=[keras.metrics.MeanAbsoluteError()])
        metrics=[keras.metrics.MeanSquaredError()])

    if training is True:
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(comp_train, raw_train,
                    batch_size=64,
                    epochs=100,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(comp_dev, raw_dev),
                    callbacks=callbacks_list)

    if plot is True:
        which_num = plot_idx
        comp_re = np.reshape(comp_train[which_num], (1,1, 71))
        predicted = model.predict(comp_re)
        predicted = predicted.reshape((71,1))
        plt.figure()
        plt.title('Compressed...')
        plt.plot(comp_arr_norm_backup[which_num])
        if save is True:
            plt.savefig(f"../img/compressed_{which_num}.png")
        plt.figure()
        plt.title('Recon...')
        plt.plot(predicted)
        if save is True:
            plt.savefig(f"../img/reconstructed_{which_num}.png")
        plt.figure()
        plt.title('Raw...')
        plt.plot(raw_arr_norm_backup[which_num])
        if save is True:
            plt.savefig(f"../img/actual_{which_num}.png")
        if show is True:
            plt.show()

if __name__=="__main__":
    train(
        use_chk=True, 
        plot=True, 
        show=True,
        save=False,
        plot_idx=24)


# if __name__=="__main__":
#     train(
#         use_chk=False, 
#         plot=True, 
#         show=True,
#         save=False,
#         plot_idx=75)