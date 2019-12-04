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

def get_stft(arr):
    arr = tf.convert_to_tensor(arr, dtype=tf.float32)
    arr_flat = tf.reshape(arr, [-1])
    arr_pad = tf.pad(arr_flat, [[3,4]])     #stft takes in T
    arr_stft = tf.abs(tf.signal.stft(arr_pad, frame_length=8, frame_step=1, fft_length=8))    #stft outputs T x F
    arr_stft = tf.transpose(arr_stft, perm=[1,0])  #rearrange to F x T
    return arr_stft

def plot_results(plot_idx, example, model, save=False, show=True):
    example_comp = example[0][plot_idx].numpy().reshape((1,71,1))
    example_raw = example[1]['1d_output'][plot_idx].numpy()
    example_stft = example[1]['stft_output'][plot_idx].numpy()
    (predicted_1D, predicted_stft) = model.predict(example_comp)
    
    plt.figure(figsize=(14,7))
    plt.subplot(3,2,1)
    plt.title('Compressed TD')
    plt.plot(example_comp.reshape((71,1)))
    plt.subplot(3,2,2)
    plt.title('Compressed Spectrogram')
    plt.imshow(get_stft(example_comp), origin='lower', aspect='auto')
    plt.subplot(3,2,3)
    plt.title('Reconstructed TD')
    plt.plot(predicted_1D.reshape((71,1)))
    plt.subplot(3,2,4)
    plt.title('Reconstructed Spectrogram')
    plt.imshow(predicted_stft[0,:,:,0], origin='lower', aspect='auto')
    plt.subplot(3,2,5)
    plt.title('Raw TD')
    plt.plot(example_raw.reshape((71,1)))
    plt.subplot(3,2,6)
    plt.title('Raw Spectrogram')
    plt.imshow(example_stft[:,:,0], origin='lower', aspect='auto')
    
    if save is True:
        plt.savefig("..\\img\\waveforms_{plot_idx}.png")
    if show is True:
        plt.show()

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
    raw_arr_stft = tf.transpose(raw_arr_stft, perm=[1,0])  #rearrange to B x F x T
    raw_arr_stft = tf.expand_dims(raw_arr_stft, -1)     #change to B x F x T x 1, same as Spectrogram layer output
    raw_arr_norm = raw_arr_norm[None, :]    #model outputs B x 1 x T for TD vector
    return comp_arr, {"1d_output": raw_arr_norm, "stft_output": raw_arr_stft}

def model_create(example_length):
    inputs = keras.Input(shape=(example_length,1), name="in")
    x = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same', activation='relu' ,data_format='channels_last', use_bias=True, kernel_initializer='glorot_uniform', name="conv_1")(inputs)
    x = layers.Bidirectional(layers.GRU(units=64, activation='tanh', recurrent_activation='tanh', kernel_initializer='glorot_uniform', return_sequences=True, name='biGRU_1'), merge_mode='ave')(x)
    x = layers.Dense(1024, activation="relu", name="dense_2")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dense(1024, activation="relu", name="dense_3")(x)
    x = layers.Dense(1024, activation="relu", name="dense_4")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dense(1024, activation="relu", name="dense_5")(x)
    x = layers.Dense(64, name="dense_6")(x) #(B x T x 64)
    x = layers.Conv1D(filters=1, kernel_size=8, strides=1, padding='same', activation=None, data_format='channels_last', kernel_initializer='glorot_uniform', name="conv_2")(x)    #(B x T x 1)
    x = layers.Dense(1, name="dense_7")(x)  # (B x T x 1)
    x_1d = layers.Permute((2,1), name='1d_output')(x)    # (B x 1 x T)
    x = layers.Flatten(name='flatten_1')(x_1d)  # (B x T)
    x = layers.Lambda(lambda y: tf.pad(y, [[0,0],[3,4]]), name='pad_1')(x)     #stft takes in B x T
    raw_arr_stft = layers.Lambda(lambda y: tf.abs(tf.signal.stft(y, frame_length=8, frame_step=1, fft_length=8)), name='spectrogram')(x)    #stft outputs B x T x F
    raw_arr_stft = layers.Lambda(lambda y: tf.transpose(y, perm=[0,2,1]), name='rearrange_1')(raw_arr_stft)  #change to B x F x T x 1, same as Spectrogram layer output
    x_spect = layers.Reshape((5,71,1), name='stft_output')(raw_arr_stft)
    return keras.Model(inputs=inputs, outputs=[x_1d, x_spect])
    
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
        
    model = model_create(71)
    model.summary()
    
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
    lossWeights = {"1d_output": 0.9,
                   "stft_output": 0.1}
#    model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),  # Optimizer
        # Loss function to minimize
        #loss=keras.losses.MeanAbsoluteError(),
        loss=losses, loss_weights=lossWeights,
        # List of metrics to monitor
        #metrics=[keras.metrics.MeanAbsoluteError()])
        metrics=[keras.metrics.MeanSquaredError()])
    
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
    # print(f"TOTAL NUMBER OF EXAMPLES: {total_size}")
    
    newds = tf.compat.v2.data.TFRecordDataset(final_files, compression_type="GZIP", num_parallel_reads=16)
    buffer_size = 3000000
    batch_size = 128

    val_iterator = newds.take(65536).__iter__()
    train_iterator = newds.skip(65536).__iter__()
    newds = newds.map(parse_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # newds = newds.shuffle(buffer_size)
    newds = newds.repeat()
    newds = newds.batch(batch_size, drop_remainder=True)
    newds = newds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    if training is True:
        epochs = 5
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
        iterator = newds.__iter__()
        example = iterator.get_next()
        plot_results(plot_idx, example, model, save, show)
    pdb.set_trace()
    K.clear_session()

if __name__=="__main__":
    train(
        use_chk=False, 
        plot=True, 
        show=True,
        save=False,
        plot_idx=24)