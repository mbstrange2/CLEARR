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
from pathlib import Path
import math
import random

def filter_data_files(item):
    if "Compressed" not in item:
        return True
    else:
        return False
        
tf.executing_eagerly()

def reformat_data(raw_path, comp_path, ignore_list, raw_array_path, comp_array_path):

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
        data_files = os.listdir(subfolders[qqq]) 
        subfiles = os.listdir(subfolders[qqq])
        subfiles_filtered = filter(filter_data_files, subfiles)
        for p in subfiles_filtered:
            full_path = os.path.join(subfolders[qqq], p)
           # print(full_path)
            compressed_tokens = full_path.split("data000")
            electrode_num = int(compressed_tokens[1].split("_")[2])
            if(electrode_num in ignore_list):
                print(f"ignoring electrode: {electrode_num}")
                continue

            compressed_path = compressed_tokens[0] + "data000Compressed" + compressed_tokens[1]
            compressed_path = compressed_path.replace("spikes", "spike", 1) # Small file naming error
            #print(compressed_path)
            if os.path.exists(compressed_path):
                #print(f"Processing {full_path}")
                print(f"File number: {file_num}")
                file_num = file_num + 1
                raw_inputs = loadmat(full_path)
                raw_arr = raw_inputs["spikeArr"].squeeze()
                comp_inputs = loadmat(compressed_path)
                comp_arr = comp_inputs["spikeArr"].squeeze()
                raw_input_np = np.asarray(raw_arr).T
                #print(raw_input_np.shape)
                comp_input_np = np.asarray(comp_arr).T

                if(len(raw_input_np.shape) > 1):
                    raw_input_collapsed = np.zeros((raw_input_np.shape[0], raw_input_np.shape[1]-1))
                    comp_input_collapsed = np.zeros((raw_input_np.shape[0], raw_input_np.shape[1]-1))
                #print(raw_input_collapsed.shape)

                    jj = 0
                    for ii in range(raw_input_np.shape[0]):
                        #if((raw_input_np[ii][0] != 0)):
                        if((raw_input_np[ii][0] != 0) and (np.ptp(comp_input_np[ii][1:]) > 10)):
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

            else:
                print("Raw path exists, but compressed path does: " + str(full_path))

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
    #comp_arr_norm = tf.math.l2_normalize(comp_arr)
    raw_arr_norm = tf.divide(tf.subtract(raw_arr, tf.reduce_min(raw_arr)), 
                            tf.subtract(tf.reduce_max(raw_arr), tf.reduce_min(raw_arr))
                        )
    return comp_arr, raw_arr_norm

def train(remake=False, use_chk=False, plot=False, show=False, save=False, plot_idx=0):

    raw_path = Path("C:/Users/Mek/DATA_TAR") 
    comp_path = Path("C:/Users/Mek/DATA_TAR")
    ignore_list = [45, 49, 177, 401, 418, 460]

    raw_array_path = "./data_array"
    comp_array_path = "./comp_array"

    if not os.path.exists(raw_array_path + "_88.npz")  or remake is True:
        print("np files don't exist, creating now...")
        reformat_data(raw_path, comp_path, ignore_list, raw_array_path, comp_array_path)
    else:
        print("NP array files already exist, proceeding as usual...")

    filenames = []
    num_train = 84
    # convert to TFRecord
    if os.path.exists("./file_sizes.npy") is False:
        print("Creating file array...")
        size_arr = np.zeros(num_train)
        for i in range(num_train):
            #if(i == 3):
            #    break
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

    num_use = 3
    #file_idx = list(range(num_train))
    # Shuffle the files to prevent grouping
    #random.shuffle(file_idx)
    file_idx = [54, 10, 32]
    final_files = []
    total_size = 0
    for i in range(num_use):
        final_files.append(f"tf_records_{file_idx[i]}.tfrecordz")
        # Accumulate the number of examples in each file
        total_size += int(size_arr[file_idx[i]])

    print(f"Files for training: {final_files}")
    print(f"TOTAL NUMBER OF EXAMPLES: {total_size}")

    newds = tf.compat.v2.data.TFRecordDataset(final_files, compression_type="GZIP")
    buffer_size = 3000000
    batch_size = 256

    newds = newds.map(parse_proto)
    newds = newds.shuffle(buffer_size)
    newds = newds.repeat()
    newds = newds.batch(batch_size, drop_remainder=True)
    #ds_it = iter(newds)

    def model_create(example_length):
        inputs = keras.Input(shape=(example_length, ), name="in")
        x = layers.Dense(2000, activation="relu", name="dense_1")(inputs)
        #x = layers.Conv1D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu' ,data_format='channels_first', use_bias=True, kernel_initializer='glorot_uniform')(inputs)
     #   x = layers.Conv1D(filters=50, kernel_size=5, strides=2, padding='same', activation=None,data_format='channels_first', use_bias=True, kernel_initializer='glorot_uniform')(x)
      #  x = layers.Dense(1000, activation="relu", name="dense_1")(inputs)
      #  x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.Dense(2000, activation="relu", name="dense_2")(x)
        x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.Dense(2000, activation="relu", name="dense_3")(x)
        x = layers.Dense(2000, activation="relu", name="dense_4")(x)
       # x = layers.Dense(1000, activation="relu", kernel_regularizer=keras.regularizers.l2(0.002), name="dense_4")(x)
        x = layers.BatchNormalization(name="batch_norm_2")(x)
        x = layers.Dense(2000, activation="relu", name="dense_5")(x)
        #x = layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation=None,data_format='channels_first', use_bias=True, kernel_initializer='glorot_uniform')(inputs)
    #    x = layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation=None, data_format='channels_first', kernel_initializer='glorot_uniform')(x)
        #outputs = layers.Dense(example_length, activation="tanh", name="out")(x)
        outputs = layers.Dense(example_length, name="out")(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    model = model_create(71)

    checkpoint_filepath = "./model_chk.hdf5"
    training = False

    if os.path.exists(checkpoint_filepath) and use_chk is True:
        print("Reading checkpointed model...")
        model.load_weights(checkpoint_filepath)
    else:
        print("Training model...")
        training = True

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.MeanSquaredError(),
        # List of metrics to monitor
        metrics=[keras.metrics.MeanSquaredError()])


    STEPS_PER_EPOCH = math.floor(total_size / batch_size) - 100
    #print(f"STEPS PER EPOCH: {STEPS_PER_EPOCH}")

    if training is True:
       # batch_size = 128
        epochs = 15
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(newds,
                   # batch_size=batch_size, # 128
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=epochs, # 100
                    # Let 20% of the training data be used for 
                    # our deveset
                  #  validation_split=.15,
                    # Have this callback for checkpointing our model
                    callbacks=callbacks_list)
        hist_loss = history.history['loss']
        plt.figure()
        plt.title('Loss vs epoch...')
        plt.plot(hist_loss)
        plt.savefig(f"./img/loss_batch_size_{batch_size}_epochs_{epochs}.png")
        if show is True:
            plt.show()

    newds_iter = iter(newds)
    example = next(newds_iter)
    #print(example)
    example_comp = example[0][0]
    example_raw = example[1][0]
    #print(example_comp[0])
    #print(example_raw)

    # TODO: Fix this code
    if plot is True:

        #which_num = plot_idx
        comp_re = np.reshape(example_comp, (1, 71))
        #print(example_comp.shape)
        predicted = model.predict(comp_re)
        predicted = predicted.reshape((71,1))
        plt.figure()
        plt.title('Compressed...')
        plt.plot(example_comp)
        if save is True:
            plt.savefig(f"../img/compressed_example.png")
            #plt.savefig(f"../img/compressed_example{which_num}.png")
        plt.figure()
        plt.title('Recon...')
        plt.plot(predicted)
        if save is True:
            plt.savefig(f"../img/reconstructed_example.png")
            #plt.savefig(f"../img/reconstructed_{which_num}.png")
        plt.figure()
        plt.title('Raw...')
        plt.plot(example_raw)
        #plt.plot(raw_arr_norm_backup[which_num])
        if save is True:
            plt.savefig(f"../img/actual_example.png")
            #plt.savefig(f"../img/actual_{which_num}.png")
        if show is True:
            plt.show()

if __name__=="__main__":
    train(
        remake=False,
        use_chk=False, 
        plot=True, 
        show=True,
        save=False,
        plot_idx=25016)

   # raw, comp = next(ds_it)

    #print(raw)
    #print(comp)

   # return

   # raw_arr = np.load(raw_array_path)
   # comp_arr = np.load(comp_array_path)
    
   # n_x = len(raw_arr[:,0])

   # raw_arr_norm = tf.keras.utils.normalize(raw_arr, axis=1)
  #  raw_arr_norm = raw_arr
  #  comp_arr_norm = comp_arr / 255
#
  #  raw_arr_norm = raw_arr_norm.T
 #   comp_arr_norm = comp_arr_norm.T

 #   raw_arr_norm_backup = raw_arr_norm
 #   comp_arr_norm_backup = comp_arr_norm

  #  raw_arr_norm = raw_arr_norm[:, None, :]
 #   comp_arr_norm = comp_arr_norm[:, None, :]

  #  comp_train = comp_arr_norm
 #   raw_train = raw_arr_norm





    #raw_files = listdir(raw_path)
    #comp_files = listdir(comp_path)

    # raw_dict = {}
    # comp_dict = {}

    # for f in raw_files:
    #     electrode_num = None
    #     if f.endswith('.mat'):
    #         temp_inputs = loadmat(raw_path / str(f))
    #         f = f.split(".")[0] # Get rid of file extension
    #         print("Processing " + str(f))
    #         electrode_num = int(f.split("_")[2])
    #         spike_num = int(f.split("_")[4])
    #         spikedness = f.split("_")[3]
    #         input_arr = temp_inputs['temp']
    #         if electrode_num not in ignore_list:
    #             raw_dict[(electrode_num, spike_num, spikedness)] = input_arr.squeeze()

    # for f in comp_files:
    #     electrode_num = None
    #     if f.endswith('.mat'):
    #         temp_inputs = loadmat(comp_path / str(f))
    #         f = f.split(".")[0] # Get rid of file extension
    #         print("Processing " + str(f))
    #         electrode_num = int(f.split("_")[2])
    #         spike_num = int(f.split("_")[4])
    #         spikedness = f.split("_")[3]
    #         input_arr = temp_inputs['temp']
    #         if electrode_num not in ignore_list:
    #             comp_dict[(electrode_num, spike_num, spikedness)] = input_arr.squeeze()

    # num_examples = len(raw_dict.keys())

    # print("Number of examples: " + str(num_examples))

    # length_example = len(raw_dict[(1,1, "nonspike")])
    # length_example = len(raw_dict[(1,1, "spike")])

    # raw_arr = np.zeros((length_example, num_examples))
    # comp_arr = np.zeros((length_example, num_examples))

    # i = 0
    # for (elec_num, spike_num, spikedness) in raw_dict.keys():
    #     #print("Electrode: " + str(elec_num) + "\tSpike: " + str(spike_num))
    #     if (elec_num, spike_num, spikedness) not in comp_dict.keys():
    #         print("FAILURE...not in comp_dict")
    #         break
    #     raw_arr[:,i] = raw_dict[(elec_num, spike_num, spikedness)]
    #     comp_arr[:,i] = comp_dict[(elec_num, spike_num, spikedness)]
    #     i += 1

    # np.save(raw_array_path, raw_arr)
    # np.save(comp_array_path, comp_arr)