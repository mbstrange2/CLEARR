from numba import jit, cuda
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
#from tensorflow import keras
#from tensorflow.keras import layers
from kapre.time_frequency import Spectrogram
import keras as keras
from keras import layers
from keras.models import Sequential

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


def train(use_chk=False, plot=False, show=False, save=False, plot_idx=0):
    raw_path = "D:\\Programs\\Python\\Projects\\CS230\\data\\mat_2013-05-28-4\\raw\\" # Change this for yourself
    comp_path = "D:\\Programs\\Python\\Projects\\CS230\data\\mat_2013-05-28-4\\compressed\\" # Change this for yourself
    ignore_list = [45, 49, 177, 401, 418, 460]

    raw_array_path = ".\\raw_array.npy"
    comp_array_path = ".\\comp_array.npy"

    if not os.path.exists(raw_array_path) or not os.path.exists(comp_array_path):
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
    #comp_arr_norm = comp_arr / 255
    comp_arr_norm = comp_arr
    
    raw_arr_norm_kapre = np.expand_dims(raw_arr.T, -1)   ## (B x T) -> (B x T x 1)
    comp_arr_norm_kapre = np.expand_dims(comp_arr.T, -1)   ## (B x T) -> (B x T x 1)
    raw_arr_norm = np.expand_dims(raw_arr.T, 1)
    comp_arr_norm = np.expand_dims(comp_arr.T, 1)
    
    raw_train_kapre = raw_arr_norm_kapre[0:50000]
    raw_dev_kapre = raw_arr_norm_kapre[50000:]
    comp_train_kapre = comp_arr_norm_kapre[0:50000]
    comp_dev_kapre = comp_arr_norm_kapre[50000:]
    # raw_train = tf.dtypes.cast(raw_arr_norm[0:12000], tf.float32)
    # raw_dev = tf.dtypes.cast(raw_arr_norm[12000:], tf.float32)
    # comp_train = tf.dtypes.cast(comp_arr_norm[0:12000], tf.float32)
    # comp_dev = tf.dtypes.cast(comp_arr_norm[12000:], tf.float32)
    raw_train = raw_arr_norm[0:50000]
    raw_dev = raw_arr_norm[50000:]
    comp_train = comp_arr_norm[0:50000]
    comp_dev = comp_arr_norm[50000:]
    
    print("raw_train_kapre shape = ", raw_train_kapre.shape)
    print("raw_dev_kapre shape = ", raw_dev_kapre.shape)

    def model_create(example_length):
        inputs = keras.Input(shape=(example_length,1), name="in")
        x = layers.Dense(1500, activation="relu", name="dense_1")(inputs)
        x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.Dense(1500, activation="relu", name="dense_2")(x)
        x = layers.Dense(1500, activation="relu", name="dense_3")(x)
        x = layers.BatchNormalization(name="batch_norm_2")(x)
        x = layers.Dense(1500, activation="relu", name="dense_4")(x)
        x = layers.Dense(1500, activation="relu", name="dense_5")(x)
        #outputs = layers.Dense(example_length, activation="tanh", name="out")(x)
        x = layers.Dense(example_length, name="dense_6")(x) #(B x T x T)
        x = layers.Dense(1, name="dense_7")(x)  # (B x T x 1)
        x_1d = layers.Permute((2,1), name='1d_output')(x)    # (B x 1 x T) because this is format that kapre wants
        x_spect = Spectrogram(n_dft=8, n_hop=1, padding='same', power_spectrogram=1.0,
                              return_decibel_spectrogram=False, trainable_kernel=False,
                              image_data_format='channels_last', name='stft_output')(x_1d)  # (B x F x T x 1)
        return keras.Model(inputs=inputs, outputs=[x_1d, x_spect])
    
    def model_create_spectrogram(example_length):
        inputs = keras.Input(shape=(example_length,1), name="in")
        x = layers.Permute((2,1))(inputs)    # (B x 1 x T) because this is format that kapre wants
        outputs = Spectrogram(n_dft=8, n_hop=1, padding='same', power_spectrogram=1.0,
                              return_decibel_spectrogram=False, trainable_kernel=False,
                              image_data_format='channels_last', name='trainable_stft')(x)  # (B x F x T x 1)
        return keras.Model(inputs=inputs, outputs=outputs)
        
    model = model_create(n_x)
    model.summary()
    
    model_spectrogram = model_create_spectrogram(n_y)
    model_spectrogram.summary()
    
    checkpoint_filepath = ".\\model_chk.hdf5"
    training = False

    if os.path.exists(checkpoint_filepath) and use_chk is True:
        print("Reading checkpointed model...")
        model.load_weights(checkpoint_filepath)
    else:
        print("Training model...")
        training = True

    losses = {"1d_output": "mean_squared_error",
              "stft_output": "mean_squared_error"}
    lossWeights = {"1d_output": 0.3,
                   "stft_output": 0.7}
#    model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
    model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
        # Loss function to minimize
        #loss=keras.losses.MeanAbsoluteError(),
        loss=losses, loss_weights=lossWeights,
        # List of metrics to monitor
        #metrics=[keras.metrics.MeanAbsoluteError()])
        metrics=[keras.metrics.MeanSquaredError()])
        
    model_spectrogram.compile(optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError())
        
    # predicted_stft = model_spectrogram.predict(raw_arr_norm)
    # print("predicted stft shape = ", predicted_stft.shape)
    
    # (predicted_1d, predicted_stft2) = model.predict(comp_arr_norm[0:5])
    # print("predicted 1d shape = ", predicted_1d.shape)
    # print("predicted stft shape = ", predicted_stft2.shape)
    
    raw_train_stft = model_spectrogram.predict(raw_train_kapre)
    raw_dev_stft = model_spectrogram.predict(raw_dev_kapre)
    
    if training is True:
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(comp_train_kapre,
                    {"1d_output": raw_train, "stft_output": raw_train_stft},
                    batch_size=64,
                    epochs=100,   
                    validation_data=(comp_dev_kapre,  # We pass some validation for monitoring validation loss and metrics at the end of each epoch
                        {"1d_output": raw_dev, "stft_output": raw_dev_stft}), 
                    callbacks=callbacks_list)
        model.save_weights(".\\model_chk_Mel.hd5")

    if plot is True:
        which_num = plot_idx
        (predicted, _) = model.predict(comp_train[which_num])
        predicted = predicted.reshape((71,1))
        actual = raw_train[which_num]
        plt.figure()
        plt.title('Compressed...')
        plt.plot(comp_train[which_num])
        if save is True:
            plt.savefig(f"..\\img\\compressed_{which_num}.png")
        plt.figure()
        plt.title('Recon...')
        plt.plot(predicted)
        if save is True:
            plt.savefig(f"..\\img\\reconstructed_{which_num}.png")
        plt.figure()
        plt.title('Raw...')
        plt.plot(actual)
        if save is True:
            plt.savefig(f"..\\img\\actual_{which_num}.png")
        if show is True:
            plt.show()

if __name__=="__main__":
    train(
        use_chk=False, 
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