from scipy.io import loadmat
import h5py
import numpy as np
from os import listdir
import pdb
from pathlib import Path
import math
import random
from decimal import *

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

def filter_data_files(item):
    if "Compressed" not in item:
        return True
    else:
        return False

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def create_test_set(raw_path, comp_path, ignore_list, array_path):
    # Put the spikes and electrode number in here...
    comp_list = []

    final_set = 0
    file_num = 1
    zzz = 0

    check_num = 0

    subfolders = [f.path for f in os.scandir(raw_path) if f.is_dir()]   
    #print(str(subfolders))
    for qqq in range(len(subfolders)):
        print(f"FOLDER: {qqq}")
        data_files = os.listdir(subfolders[qqq]) 
        subfiles = os.listdir(subfolders[qqq])
        subfiles_filtered = filter(filter_data_files, subfiles)
        for p in subfiles_filtered:

            full_path = os.path.join(subfolders[qqq], p)
            compressed_tokens = full_path.split("data000")
            electrode_num = int(compressed_tokens[1].split("_")[2])
            if(electrode_num in ignore_list):
                print(f"ignoring electrode: {electrode_num}")
                continue

            compressed_path = compressed_tokens[0] + "data000Compressed" + compressed_tokens[1]
            compressed_path = compressed_path.replace("spikes", "spike", 1) # Small file naming error
            starting = int(compressed_tokens[1].split("_")[8].split(".")[0].split("m")[0])

            if (starting != 0) and (starting != 5):
                continue

            if os.path.exists(compressed_path):
                print(f"File number: {file_num}")
                file_num = file_num + 1
                comp_inputs = loadmat(compressed_path)
                comp_arr = comp_inputs["spikeArr"]
                comp_input_np = np.asarray(comp_arr).T

                if(len(comp_input_np.shape) > 1):
                    comp_input_collapsed = np.zeros((comp_input_np.shape[0], comp_input_np.shape[1]+1))

                    jj = 0
                    for ii in range(comp_input_np.shape[0]) :
                        if (comp_input_np[ii][0] != 0) and (np.ptp(comp_input_np[ii, 1:]) > 0): 

                            spike_num = comp_input_np[ii][0]
                            comp_input_collapsed[jj][0] = electrode_num
                            comp_input_collapsed[jj][1] = spike_num
                            comp_input_collapsed[jj][2:] = comp_input_np[ii][1:]
                            #print(f"Spike num = {spike_num}, electrode = {electrode_num}, spike is {comp_input_np[ii][1:]}")
                            jj = jj + 1

                    comp_input_collapsed = comp_input_collapsed[0:jj, :]
                    comp_list.append(comp_input_collapsed)

                    if(len(comp_list) == 100):
                        len_sum = 0
                        lens = []

                        for i in range(len(comp_list)):
                            temp_len = comp_list[i].shape[0]
                            len_sum = len_sum + temp_len
                            lens.append(temp_len)

                        final_np_comp = np.zeros((len_sum, 73))

                        tracker = 0
                        for i in range(len(comp_list)):
                            if(lens[i] == 0):
                                continue

                            final_np_comp[tracker:tracker+lens[i], :] = comp_list[i]
                            tracker = tracker + lens[i]

                        if(os.path.exists(array_path + f"_{check_num}.npz")):
                            print("Don't need to remake this file...")
                        else:
                            # first array is the raw, second is the compressed
                            np.savez_compressed(array_path + f"_{check_num}.npz", final_np_comp)

                        check_num = check_num + 1
                        comp_list = []
                else:
                    print("Skipping for other reasons...")
    # One last one
    if(len(comp_list) > 0):
        len_sum = 0
        lens = []

        for i in range(len(comp_list)):
            temp_len = comp_list[i].shape[0]
            #print(f"LEN {i} : {temp_len}")
            len_sum = len_sum + temp_len
            lens.append(temp_len)

        final_np_comp = np.zeros((len_sum, 73))

        tracker = 0
        for i in range(len(comp_list)):
            if(lens[i] == 0):
                continue

            final_np_comp[tracker:tracker+lens[i], :] = comp_list[i]
            tracker = tracker + lens[i]

        if(os.path.exists(array_path + f"_{check_num}_test.npz")):
            print("Don't need to remake this file...")
        else:
            # first array is the raw, second is the compressed
            np.savez_compressed(array_path + f"_{check_num}_test.npz", final_np_comp)


def reformat_data(raw_path, comp_path, ignore_list, raw_array_path, comp_array_path):
    print_now = 1
    # Put the spikes and electrode number in here...
    raw_list = []
    comp_list = []

    final_set = 0
    file_num = 1
    zzz = 0

    check_num = 0

    subfolders = [f.path for f in os.scandir(raw_path) if f.is_dir()]   
    #print(str(subfolders))
    for qqq in range(len(subfolders)):
        print(f"FOLDER: {qqq}")
        data_files = os.listdir(subfolders[qqq]) 
        subfiles = os.listdir(subfolders[qqq])
        subfiles_filtered = filter(filter_data_files, subfiles)
        for p in subfiles_filtered:

            full_path = os.path.join(subfolders[qqq], p)
            compressed_tokens = full_path.split("data000")
            electrode_num = int(compressed_tokens[1].split("_")[2])
            if(electrode_num in ignore_list):
                print(f"ignoring electrode: {electrode_num}")
                continue

            compressed_path = compressed_tokens[0] + "data000Compressed" + compressed_tokens[1]
            compressed_path = compressed_path.replace("spikes", "spike", 1) # Small file naming error

            starting = int(compressed_tokens[1].split("_")[8].split(".")[0].split("m")[0])

            if (starting == 0) or (starting == 5):
                continue

            if os.path.exists(compressed_path):
                print(f"File number: {file_num}")
                file_num = file_num + 1
                raw_inputs = loadmat(full_path)
                raw_arr = raw_inputs["spikeArr"]
                comp_inputs = loadmat(compressed_path)
                comp_arr = comp_inputs["spikeArr"]
                raw_input_np = np.asarray(raw_arr).T
                comp_input_np = np.asarray(comp_arr).T

                if(len(raw_input_np.shape) > 1):
                    raw_input_collapsed = np.zeros((raw_input_np.shape[0], raw_input_np.shape[1]-1))
                    comp_input_collapsed = np.zeros((raw_input_np.shape[0], raw_input_np.shape[1]-1))

                    jj = 0
                    for ii in range(raw_input_np.shape[0]):
                        if (raw_input_np[ii][0] != 0) and (np.ptp(comp_input_np[ii, 1:]) > 0):
                            raw_input_collapsed[jj] = raw_input_np[ii][1:]
                            comp_input_collapsed[jj] = comp_input_np[ii][1:]
                            jj = jj + 1

                    raw_input_collapsed = raw_input_collapsed[0:jj, :]
                    comp_input_collapsed = comp_input_collapsed[0:jj, :]

                    raw_list.append(raw_input_collapsed)
                    comp_list.append(comp_input_collapsed)

                    if(len(raw_list) == 100):
                        len_sum = 0
                        lens = []

                        for i in range(len(raw_list)):
                            temp_len = raw_list[i].shape[0]
                            #print(f"LEN {i} : {temp_len}")
                            len_sum = len_sum + temp_len
                            lens.append(temp_len)

                        final_np_raw = np.zeros((len_sum, 71))
                        final_np_comp = np.zeros((len_sum, 71))

                        tracker = 0
                        for i in range(len(raw_list)):
                            if(lens[i] == 0):
                                continue

                            final_np_raw[tracker:tracker+lens[i], :] = raw_list[i]
                            final_np_comp[tracker:tracker+lens[i], :] = comp_list[i]
                            tracker = tracker + lens[i]

                        if(os.path.exists(raw_array_path + f"_{check_num}.npz")):
                            print("Don't need to remake this file...")
                        else:
                            # first array is the raw, second is the compressed
                            np.savez_compressed(raw_array_path + f"_{check_num}.npz", (final_np_raw, final_np_comp))

                        check_num = check_num + 1
                        raw_list = []
                        comp_list = []

    if(len(raw_list) > 0):
        len_sum = 0
        lens = []

        for i in range(len(raw_list)):
            temp_len = raw_list[i].shape[0]
            len_sum = len_sum + temp_len
            lens.append(temp_len)

        final_np_raw = np.zeros((len_sum, 71))
        final_np_comp = np.zeros((len_sum, 71))

        tracker = 0
        for i in range(len(raw_list)):
            if(lens[i] == 0):
                continue

            final_np_raw[tracker:tracker+lens[i], :] = raw_list[i]
            final_np_comp[tracker:tracker+lens[i], :] = comp_list[i]
            tracker = tracker + lens[i]

        if(os.path.exists(raw_array_path + f"_{check_num}.npz")):
            print("Don't need to remake this file...")
        else:
            # first array is the raw, second is the compressed
            np.savez_compressed(raw_array_path + f"_{check_num}.npz", (final_np_raw, final_np_comp))

# Convert numpyz array to tfrecord files
def array_to_tfrecords(comp, raw, output_file):
    with tf.io.TFRecordWriter(output_file, options="GZIP") as writer:
        for j in range(comp.shape[0]):
            comp_row = comp[j]
            raw_row = raw[j]
            features = {
                'comp': tf.train.Feature(float_list=tf.train.FloatList(value=comp_row.flatten())),
                'raw': tf.train.Feature(float_list=tf.train.FloatList(value=raw_row.flatten()))
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            serialized = example.SerializeToString()
            writer.write(serialized)

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
    x1 = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same', activation='relu' ,data_format='channels_last', use_bias=True, kernel_initializer='glorot_uniform', name="conv_1d")(inputs)
    x1 = layers.Bidirectional(layers.GRU(units=64, activation='tanh', recurrent_activation='tanh', kernel_initializer='glorot_uniform', return_sequences=True, name='biGRU_1'), merge_mode='ave')(x1)
    x1 = layers.Dense(1024, activation="relu", name="dense_2")(x1)
    x1 = layers.BatchNormalization(name="batch_norm_1")(x1)
    x1 = layers.Dense(1024, activation="relu", name="dense_3")(x1)
    x1 = layers.Dense(1024, activation="relu", name="dense_4")(x1)
    x1 = layers.BatchNormalization(name="batch_norm_2")(x1)
    x1 = layers.Dense(1024, activation="relu", name="dense_5")(x1)
    x1 = layers.Dense(64, name="dense_6")(x1) #(B x T x 64)
    x1 = layers.Dense(1, name="dense_7")(x1)  # (B x T x 1)
    x1 = layers.Permute((2,1), name='permute1')(x1)    # (B x 1 x T)
    
    x2 = layers.Flatten(name='flatten_1')(inputs)  # (B x T)
    x2 = layers.Lambda(lambda y: tf.pad(y, [[0,0],[3,4]]), name='pad_1')(x2)     #stft takes in B x T
    x2 = layers.Lambda(lambda y: tf.abs(tf.signal.stft(y, frame_length=8, frame_step=1, fft_length=8)), name='initial_stft')(x2)    #stft outputs B x T x F
    x2 = layers.Reshape((71,5,1), name='reshape_1')(x2)
    x2 = layers.Conv2D(filters=16, kernel_size=(8,3), strides=1, padding='same', activation='relu', data_format='channels_last', use_bias=True, kernel_initializer='glorot_uniform', name="conv2d_1")(x2)
    x2 = layers.Conv2D(filters=16, kernel_size=(8,3), strides=1, padding='same', activation='relu', data_format='channels_last', use_bias=True, kernel_initializer='glorot_uniform', name="conv2d_2")(x2)
    x2 = layers.AveragePooling2D(pool_size=2, strides=1, padding='same', name='mpool_1')(x2)
    x2 = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='relu', data_format='channels_last', use_bias=True, kernel_initializer='glorot_uniform', name="conv1d_1")(x2)
    x2 = layers.Conv2D(filters=16, kernel_size=(8,3), strides=1, padding='same', activation='relu', data_format='channels_last', use_bias=True, kernel_initializer='glorot_uniform', name="conv2d_3")(x2)
    x2 = layers.Conv2D(filters=16, kernel_size=(8,3), strides=1, padding='same', activation='relu', data_format='channels_last', use_bias=True, kernel_initializer='glorot_uniform', name="conv2d_4")(x2)
    x2 = layers.AveragePooling2D(pool_size=2, strides=1, padding='same', name='mpool_2')(x2)
    x2 = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='relu', data_format='channels_last', use_bias=True, kernel_initializer='glorot_uniform', name="x_spect")(x2)
    x_spect = layers.Reshape((5,71,1), name='reshape_stft_output')(x2)
    x_spect = layers.Lambda(lambda y: tf.abs(y), name='stft_output')(x_spect)   # (B x F x T x 1)
    x2 = layers.Reshape((71,5), name='reshape_stft_inverse')(x2)
    x2 = layers.Lambda(lambda y: tf.signal.inverse_stft(tf.dtypes.complex(y,y*0), frame_length=1, frame_step=1, fft_length=8), name='inverse_stft')(x2)
    x2 = layers.Reshape((1,71), name='reshape_2')(x2) # (B x 1 x T)
    
    # x_1d = layers.Average(name='1d_output')([x1, x2])  # (B x 1 x T)
    x_1d = layers.Add(name='1d_output')([x1, x2])  # (B x 1 x T)
    return keras.Model(inputs=inputs, outputs=[x_1d, x_spect])
    
def train(remake=False, use_chk=False, make_test=False, plot=False, show=False, save=False, epochs=1, plot_idx=0, val=False):
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
    pdb.set_trace()
    checkpoint_filepath = ".\\model_chk.hdf5"
    training = False

    if os.path.exists(checkpoint_filepath) and use_chk is True:
        print("Reading checkpointed model...")
        model.load_weights(checkpoint_filepath)
    else:
        print("Training model...")
        training = True
    
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
    
    ##### TRAINING HYPERPARAMETERS #####
    l_r = .001 # Default .001
    b_1 = 0.9 # Default 0.9
    b_2 = 0.999 # Default 0.999
    batch_size = 256
    ####################################
    
    newds = tf.compat.v2.data.TFRecordDataset(final_files, compression_type="GZIP", num_parallel_reads=16)
    buffer_size = 2000000

    # val_iterator = newds.take(65536).__iter__()
    # train_iterator = newds.skip(65536).__iter__()
    newds = newds.map(parse_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # newds = newds.shuffle(buffer_size)
    newds = newds.repeat()
    newds = newds.batch(batch_size) #, drop_remainder=True)
    newds = newds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    num_train_samples = 400000

    train_set = newds.take(num_train_samples)
    
    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    
    losses = {"1d_output": "mean_squared_error",
              "stft_output": "mean_squared_error"}
    lossWeights = {"1d_output": 0.5,
                   "stft_output": 0.5}
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=l_r,
            beta_1=b_1,
            beta_2=b_2
            ),  # Optimizer
        # Loss function to minimize
        loss=losses, loss_weights=lossWeights,
        # List of metrics to monitor
        metrics=[keras.metrics.MeanSquaredError()])
    
    STEPS_PER_EPOCH = math.floor(num_train_samples / batch_size) - 1
    final_loss = -1
    
    if training is True:
        epochs = 3
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        pdb.set_trace()
        history = model.fit(train_set,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=epochs,
                    # validation_split=0.05,
                    callbacks=callbacks_list)
        hist_loss = history.history['loss']
        final_loss = hist_loss[-1]
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
    
    dev_loss = -1
    if val is True:
    # Evaluation on dev set
        num_use = 1
        file_idx = [20]
        dev_files = []
        dev_size = 0
        for i in range(num_use):
            dev_files.append(f"D:\\Programs\\Python\\Projects\\CS230\\data\\tfrecords_new\\tf_records_{file_idx[i]}.tfrecordz")
            # Accumulate the number of examples in each file
            # dev_size += int(size_arr[file_idx[i]])

        print(f"Dev files for evaluation: {dev_files}")
        # print(f"TOTAL NUMBER OF EXAMPLES: {dev_size}")

        dev = tf.compat.v2.data.TFRecordDataset(dev_files, compression_type="GZIP", num_parallel_reads=16)
        # pdb.set_trace()
        dev = dev.map(parse_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dev = dev.batch(batch_size, drop_remainder=True)
        dev = dev.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dev_loss = model.evaluate(dev)[-1]
    
    ### Predict on all...
    print(f"Final Metrics: alpha={l_r}, beta_1={b_1}, beta_2={b_2}, batch_size={batch_size}, train_loss={final_loss}, dev_loss={dev_loss}, num_train_samples={num_train_samples}, dev_samples={dev_size}, epochs={epochs}")
    
    K.clear_session()

if __name__=="__main__":
    train(
        remake=False, # Create new npz
        use_chk=False, # Use checkpointed model (don't train again)
        make_test=False,
        plot=True, # plot the figures at the end
        show=True, # show any plots
        save=False, # save the images
        epochs=3,
        plot_idx=24, # which item to plot
        val=True
    ) 