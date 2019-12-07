from scipy.io import loadmat
from scipy.io import savemat
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
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from pathlib import Path
import math
import random
import time
from decimal import *

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
                        if (comp_input_np[ii][0] != 0): 

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

        if(os.path.exists(array_path + f"_{check_num}.npz")):
            print("Don't need to remake this file...")
        else:
            # first array is the raw, second is the compressed
            np.savez_compressed(array_path + f"_{check_num}.npz", final_np_comp)


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

def parse_proto(example_proto):
    features = {
        'comp': tf.io.FixedLenFeature([71,], tf.float32),
        'raw': tf.io.FixedLenFeature([71,], tf.float32)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    comp_arr = parsed_features['comp']
    raw_arr = parsed_features['raw']
    raw_arr_norm = raw_arr
    # Format data for CNN input format
    comp_arr = comp_arr[None, :]
    raw_arr_norm = raw_arr_norm[None, :]
    return comp_arr, raw_arr_norm


def model_create(example_length):

    # Model P1
    inputs = keras.Input(shape=(1, example_length, ), name="in")
    x = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='valid', activation='relu', 
            data_format='channels_first', use_bias=True, kernel_initializer='glorot_uniform')(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    x = layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu', data_format='channels_first', 
            use_bias=True, kernel_initializer='glorot_uniform')(x)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    x = layers.Conv1D(filters=96, kernel_size=3, strides=1, padding='valid', activation='relu', data_format='channels_first', 
        use_bias=True, kernel_initializer='glorot_uniform')(x)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    x = layers.Dense(2000, activation="relu", name="dense_4")(x)
    x = layers.Dense(1500, activation="relu", name="dense_5")(x)
    x = layers.Conv1D(filters=1, kernel_size=5, strides=1, padding='same', activation='relu', data_format='channels_first', use_bias=True, kernel_initializer='glorot_uniform')(x)

    outputs = layers.Dense(example_length, name="out")(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def train(remake=False, use_chk=False, make_test=False, plot=False, show=False, save=False, epochs=1, plot_idx=0, val=False):

    raw_path = Path("C:/Users/Mek/DATA_TAR") 
    comp_path = Path("C:/Users/Mek/DATA_TAR")
    ignore_list = [45, 49, 177, 401, 418, 460]

    raw_array_path = "./data_array"
    comp_array_path = "./comp_array"

    if remake is True:
    #if not os.path.exists(raw_array_path + "_0.npz")  or remake is True:
        print("np files don't exist, creating now...")
        reformat_data(raw_path, comp_path, ignore_list, raw_array_path, comp_array_path)
    else:
        print("NP array files already exist, proceeding as usual...")

    filenames = []
    num_train = 21
    # convert to TFRecord
    if os.path.exists("./file_sizes.npy") is False:
        print("Creating file array...")
        size_arr = np.zeros(num_train)
        for i in range(num_train):
            total_file = np.load(f"data_array_{i}.npz")
            raw_arr = total_file['arr_0'][0]
            size_arr[i] = int(raw_arr.shape[0])
            filenames.append(f"tf_records_{i}.tfrecordz")
            if(os.path.exists(f"tf_records_{i}.tfrecordz")):
                print(f"TFRecord {i} File Already Exists...")
                continue
            total_file = np.load(f"data_array_{i}.npz")
            comp_arr = total_file['arr_0'][1]
            array_to_tfrecords(comp_arr, raw_arr, f"tf_records_{i}.tfrecordz")
        np.save("./file_sizes.npy", size_arr)
    else:
        print("File array already exists, loading...")
        size_arr = np.load("./file_sizes.npy")

    num_use = 20
    file_idx = list(range(num_use)) #[0, 1] 
    final_files = []
    num_train_samples = 0
    for i in range(num_use):
        final_files.append(f"tf_records_{file_idx[i]}.tfrecordz")
        # Accumulate the number of examples in each file
        num_train_samples += int(size_arr[file_idx[i]])

    print(f"Files for training: {final_files}")
    print(f"TOTAL NUMBER OF EXAMPLES: {num_train_samples}")
    # Create the model
    model = model_create(71)
    plot_model(model, to_file="model_visualization.png", show_shapes=True)

    checkpoint_filepath = "./model_chk.hdf5"
    training = False

    if os.path.exists(checkpoint_filepath) and use_chk is True:
        print("Reading checkpointed model...")
        model.load_weights(checkpoint_filepath)
    else:
        print("Training model...")
        training = True

    ##### TRAINING HYPERPARAMETERS #####
    l_r = .002 # Default .001
    b_1 = 0.85 # Default 0.9
    b_2 = 0.999 # Default 0.999
    batch_size = 512
    ####################################

    #num_train_samples = 400000

    newds = tf.compat.v2.data.TFRecordDataset(final_files, compression_type="GZIP", num_parallel_reads=16)
    buffer_size = 2000000

    newds = newds.map(parse_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #newds = newds.shuffle(buffer_size)
    newds = newds.repeat()
    newds = newds.batch(batch_size) #, drop_remainder=True)
    newds = newds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    train_set = newds.take(num_train_samples)

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=l_r,
            beta_1=b_1,
            beta_2=b_2
            ),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.MeanSquaredError(),
        # List of metrics to monitor
        metrics=[keras.metrics.MeanSquaredError()])

    print(model.summary())
    STEPS_PER_EPOCH = math.floor(num_train_samples / batch_size) - 1

    final_loss = -1
    if training is True:
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(train_set,
                   # batch_size=batch_size, # 128
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=epochs, # 100
                    # Have this callback for checkpointing our model
                    callbacks=callbacks_list)
        hist_loss = history.history['loss']
        final_loss = hist_loss[-1]
        plt.figure()
        plt.title('Loss vs epoch...')
        plt.plot(hist_loss)
        plt.savefig(f"./img/loss_batch_size_{batch_size}_epochs_{epochs}.png")
        if show is True:
            plt.show()

    train_iter = iter(train_set)
    example = next(train_iter)
    #### PUMIAO + ANDREW - CHANGE THE SECOND INDEX
    # PIC
    pic = plot_idx
    example_comp = example[0][pic]
    example_raw = example[1][pic]

    if plot is True:
        #comp_re = np.reshape(example_comp, (1, 71))
        comp_re = np.reshape(example_comp, (1, 1, 71))
        predicted = model.predict(comp_re)
        predicted = predicted.reshape((71,1))
        plt.figure()
        plt.title('Compressed...')
        plt.plot(np.reshape(example_comp, (71, 1)))
        if save is True:
            plt.savefig(f"./img/compressed_example_{pic}.png")
        plt.figure()
        plt.title('Recon...')
        plt.plot(predicted)
        if save is True:
            plt.savefig(f"./img/reconstructed_example_{pic}.png")
        plt.figure()
        plt.title('Raw...')
        plt.plot(np.reshape(example_raw, (71,1)))
        if save is True:
            plt.savefig(f"./img/actual_example_{pic}.png")
        if show is True:
            plt.show()

    dev_loss = -1
    dev_size = 0
    if val is True:
    # Evaluation on dev set
        num_use = 1
        file_idx = [20]
        dev_files = []
        for i in range(num_use):
            dev_files.append(f"tf_records_{file_idx[i]}.tfrecordz")
            # Accumulate the number of examples in each file
            dev_size += int(size_arr[file_idx[i]])

        print(f"Dev files for evaluation: {dev_files}")
        print(f"TOTAL NUMBER OF EXAMPLES: {dev_size}")

        dev = tf.compat.v2.data.TFRecordDataset(dev_files, compression_type="GZIP", num_parallel_reads=16)
        dev = dev.map(parse_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dev = dev.batch(batch_size, drop_remainder=True)
        dev = dev.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dev_loss = model.evaluate(dev)[-1]

    ### Predict on all...
    print(f"Final Metrics: alpha={l_r}, beta_1={b_1}, beta_2={b_2}, batch_size={batch_size}, train_loss={final_loss}, dev_loss={dev_loss}, num_train_samples={num_train_samples}, dev_samples={dev_size}, epochs={epochs}")

    test_data = True
    #test_data = False

    if make_test is True:
        if test_data is True:
            create_test_set(raw_path, comp_path, ignore_list, './test_set')

        for i in range(11):
            if(os.path.exists(f"final_testing_spikes_{i}.mat")):
                continue
            total_file = np.load(f"test_set_{i}.npz")
            arr_comp = total_file['arr_0']
            spikes = arr_comp.shape[0]
            new_arr = np.copy(arr_comp)
            print(f"This many spikes...{arr_comp.shape}")
            spike_arr = arr_comp[:,2:]
            spike_arr = spike_arr[:, None ,:]
            t = time.time()
            # do stuff
            predicted = model.predict(spike_arr, verbose=0)
            elapsed = time.time() - t
            print(f"Elapsed time for predictions: {elapsed}")
            predicted = predicted.reshape((spikes, 71))
            predicted = ((predicted*1000).astype(int))/1000
            new_arr[:, 2:] = predicted #np.round(predicted, decimals=2)
            savemat(f"final_testing_spikes_{i}.mat", {'spikeRep': new_arr})

if __name__=="__main__":
    train(
        remake=False, # Create new npz
        use_chk=True, # Use checkpointed model (don't train again)
        make_test=True,
        plot=True, # plot the figures at the end
        show=True, # show any plots
        save=False, # save the images
        epochs=5,
        plot_idx=400, # which item to plot
        val=False
    ) 
