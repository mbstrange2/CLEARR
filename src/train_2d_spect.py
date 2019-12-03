# from numba import jit, cuda
from scipy.io import loadmat
import h5py
import numpy as np
from os import listdir
import pdb
from pathlib import Path
import math
import random

import os, sys, pdb, pickle
from multiprocessing import Pool
import numpy as np
#import samplerate
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import backend as K
from kapre.time_frequency import Spectrogram
# import keras as keras
# from keras import layers
# from keras.models import Sequential
# from keras import backend as K

# sess = tf.compat.v1.keras.backend.get_session()
# graph = tf.compat.v1.get_default_graph()

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

def parse_proto(example_proto):
    features = {
        'comp': tf.io.FixedLenFeature([71,], tf.float32),
        'raw': tf.io.FixedLenFeature([71,], tf.float32)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    comp_arr = parsed_features['comp']
    raw_arr = parsed_features['raw']
    #comp_arr_norm = tf.math.l2_normalize(comp_arr)
    #raw_arr_norm = tf.divide(tf.subtract(raw_arr, tf.reduce_min(raw_arr)), 
    #                        tf.subtract(tf.reduce_max(raw_arr), tf.reduce_min(raw_arr))
    #                    )
    raw_arr_norm = raw_arr
    comp_arr = comp_arr[:, None]    #model takes in B x T x 1
    raw_arr_kapre = tf.pad(raw_arr_norm, [[3,4]])     #stft takes in B x T
    raw_arr_stft = tf.abs(tf.signal.stft(raw_arr_kapre, frame_length=8, frame_step=1, fft_length=8))    #stft outputs B x T x F
    raw_arr_stft = tf.transpose(raw_arr_stft, perm=[1,0])  #change to B x F x T x 1, same as Spectrogram layer output
    raw_arr_stft = tf.expand_dims(raw_arr_stft, -1)
    raw_arr_norm = raw_arr_norm[None, :]    #model outputs B x 1 x T
    # pdb.set_trace()
    return comp_arr, {"1d_output": raw_arr_norm, "stft_output": raw_arr_stft}

# def create_inputs(newds, batch_size, model_spectrogram):
    # iterator = newds.__iter__()
    
    # raw_arr_norm, comp_arr = iterator.get_next()
    # raw_arr_norm = tf.reshape(raw_arr_norm, (batch_size,71,1))
    # raw_arr_norm_ds = tf.data.Dataset.from_tensor_slices(raw_arr_norm)
    # comp_arr = tf.data.Dataset.from_tensor_slices(tf.reshape(comp_arr, (batch_size,71,1)))
    pdb.set_trace()
    # raw_arr_stft = tf.data.Dataset.from_tensor_slices(model_spectrogram(raw_arr_norm))
    # batch_outputs = tf.data.Dataset.zip((raw_arr_norm_ds, raw_arr_stft))
    # ds_batch = tf.data.Dataset.zip(comp_arr, batch_outputs)
    # return ds_batch
    
def create_inputs(comp_arr, raw_arr):     #tensorflow forces tf_map function to not execute eagerly
    # x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1,1,71))
    
    #comp_arr_norm = tf.math.l2_normalize(comp_arr)
    #raw_arr_norm = tf.divide(tf.subtract(raw_arr, tf.reduce_min(raw_arr)), 
    #                        tf.subtract(tf.reduce_max(raw_arr), tf.reduce_min(raw_arr))
    #                    )
    raw_arr_norm = raw_arr
    comp_arr = comp_arr[None, :]
    raw_arr_norm = raw_arr_norm[None, :]
    # raw_arr_kapre = tf.reshape(raw_arr_norm, [256,1,71])
    raw_arr_stft = Spectrogram(n_dft=8, n_hop=1, padding='same', power_spectrogram=1.0,
                               return_decibel_spectrogram=False, trainable_kernel=False,
                               image_data_format='channels_last', name='stft_output')(raw_arr)  # (B x F x T x 1)
    raw_arr_stft = raw_arr_stft[None, :]
    pdb.set_trace()
    return comp_arr, {"1d_output": raw_arr_norm, "stft_output": raw_arr_stft}

def model_create(example_length):
        inputs = keras.Input(shape=(example_length,1), name="in")
        x = layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu' ,data_format='channels_last', use_bias=True, kernel_initializer='glorot_uniform', name="conv_1")(inputs)
        x = layers.Dense(1024, activation="relu", name="dense_2")(x)
        x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.Dense(1024, activation="relu", name="dense_3")(x)
        x = layers.Dense(1024, activation="relu", name="dense_4")(x)
        x = layers.BatchNormalization(name="batch_norm_2")(x)
        x = layers.Dense(1024, activation="relu", name="dense_5")(x)
        x = layers.Dense(64, name="dense_6")(x) #(B x T x 64)
        x = layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation=None, data_format='channels_last', kernel_initializer='glorot_uniform', name="conv_2")(x)    #(B x T x 1)
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
    
def train(use_chk=False, plot=False, show=False, save=False, plot_idx=0):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

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
    
    raw_train_kapre = raw_arr_norm_kapre[0:12000]
    raw_dev_kapre = raw_arr_norm_kapre[12000:]
    comp_train_kapre = comp_arr_norm_kapre[0:12000]
    comp_dev_kapre = comp_arr_norm_kapre[12000:]
    # raw_train = tf.dtypes.cast(raw_arr_norm[0:12000], tf.float32)
    # raw_dev = tf.dtypes.cast(raw_arr_norm[12000:], tf.float32)
    # comp_train = tf.dtypes.cast(comp_arr_norm[0:12000], tf.float32)
    # comp_dev = tf.dtypes.cast(comp_arr_norm[12000:], tf.float32)
    raw_train = raw_arr_norm[0:12000]
    raw_dev = raw_arr_norm[12000:]
    comp_train = comp_arr_norm[0:12000]
    comp_dev = comp_arr_norm[12000:]
    
    print("raw_train_kapre shape = ", raw_train_kapre.shape)
    print("raw_dev_kapre shape = ", raw_dev_kapre.shape)
        
    model = model_create(n_x)
    model.summary()
    pdb.set_trace()
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),  # Optimizer
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
    
    num_use = 1     #approx 1.5M examples per tf record
    #file_idx = list(range(num_train))
    # Shuffle the files to prevent grouping
    #random.shuffle(file_idx)
    file_idx = [0]#[54, 10, 32]
    final_files = []
    total_size = 0
    for i in range(num_use):
        final_files.append(f"D:\\Programs\\Python\\Projects\\CS230\\data\\tfrecords_new\\tf_records_{file_idx[i]}.tfrecordz")
        # Accumulate the number of examples in each file
        # total_size += int(size_arr[file_idx[i]])

    print(f"Files for training: {final_files}")
    print(f"TOTAL NUMBER OF EXAMPLES: {total_size}")
    
    newds = tf.compat.v2.data.TFRecordDataset(final_files, compression_type="GZIP", num_parallel_reads=16)
    buffer_size = 3000000
    batch_size = 128

    val_iterator = newds.take(65536).__iter__()
    train_iterator = newds.skip(65536).__iter__()
    newds = newds.map(parse_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    newds = newds.shuffle(buffer_size)
    newds = newds.repeat()
    newds = newds.batch(batch_size, drop_remainder=True)
    # pdb.set_trace()
    newds = newds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # iterator = newds.__iter__()
    
    # comp_arr, raw_arr_norm = iterator.get_next()
    # raw_arr_norm = tf.reshape(raw_arr_norm, (batch_size,1,71)).numpy()  #model outputs a B x 1 x T array
    # comp_arr = tf.reshape(comp_arr, (batch_size,71,1)).numpy()          #model takes in a B x T x 1 array
    # raw_arr_kapre = tf.reshape(raw_arr_norm, (batch_size,71,1)) #model_spectrogram wants a B x T x 1 array
    # raw_arr_stft = model_spectrogram.predict(raw_arr_kapre, steps=1)
    # pdb.set_trace()
    
    if plot is True:    #Check spectrograms and waveforms to make sure they make sense
        which_num = plot_idx
        plt.figure()
        plt.subplot(3,1,1)
        predicted = model_spectrogram.predict(raw_train_kapre[which_num].reshape((1,71,1)))
        print("Predicted shape = ", predicted.shape)
        plt.imshow(predicted[0,:,:,0], interpolation='nearest', origin='lower')
        plt.title('Kapre Predicted Spectrogram')
        plt.subplot(3,1,2)
        plt.plot(raw_train_kapre[which_num].reshape((71,1)))
        plt.title('Raw Waveform')
        plt.subplot(3,1,3)
        padded_signal = np.pad(raw_train_kapre[which_num].flatten(), pad_width=(3,4))
        orig_spectrogram, _, _, _ = plt.specgram(padded_signal, NFFT=8, Fs=20000, noverlap=7)
        plt.imshow(orig_spectrogram, interpolation='nearest', origin='lower')
        plt.title('Python Spectrogram')
        plt.show()
        #input("Press enter to continue...")
    
    if training is True:
        epochs = 3
        STEPS_PER_EPOCH = 256
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        pdb.set_trace()
        history = model.fit(newds,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=epochs,
                    # validation_split=0.05,
                    callbacks=callbacks_list)
        hist_loss = history.history['loss']
        plt.figure()
        plt.title('Loss vs epoch...')
        plt.plot(hist_loss)
        plt.savefig(f"D:\\Programs\\Python\\Projects\\CS230\\data\\loss_batch_size_{batch_size}_epochs_{epochs}.png")
        if show is True:
            plt.show()
        model.save_weights(".\\model_chk_Mel.hd5")

    if plot is True:
        newds_iter = iter(newds)
        example = next(newds_iter)
        example_comp = example[0][plot_idx].numpy().reshape((1,71,1))
        example_raw = example[1]['1d_output'][plot_idx].numpy()
        example_stft = example[1]['stft_output'][plot_idx].numpy()
        
        (predicted, _) = model.predict(example_comp)
        print("Predicted shape = ", predicted.shape)
        predicted = predicted.reshape((71,1))
        
        plt.figure()
        plt.title('Compressed...')
        plt.plot(example_comp.reshape((71,1)))
        if save is True:
            plt.savefig("..\\img\\compressed_{which_num}.png")
        plt.figure()
        plt.title('Recon...')
        plt.plot(predicted)
        if save is True:
            plt.savefig("..\\img\\reconstructed_{which_num}.png")
        plt.figure()
        plt.title('Raw...')
        plt.plot(example_raw.reshape((71,1)))
        if save is True:
            plt.savefig("..\\img\\actual_{which_num}.png")
        if show is True:
            plt.show()
        # pdb.set_trace()
        
        
        K.clear_session()

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