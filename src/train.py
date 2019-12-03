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

            if (starting != 20) and (starting != 25):
                continue
            #print(f"starting_minute is {starting}, electrode is {electrode_num}")
            #continue


            if os.path.exists(compressed_path):
                print(f"File number: {file_num}")
                file_num = file_num + 1
                comp_inputs = loadmat(compressed_path)
                comp_arr = comp_inputs["spikeArr"]
                comp_input_np = np.asarray(comp_arr).T

                if(len(comp_input_np.shape) > 1):
                    comp_input_collapsed = np.zeros((comp_input_np.shape[0], comp_input_np.shape[1]+1))

                    jj = 0
                    for ii in range(comp_input_np.shape[0]):
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

                        check_num = check_num + 1
                        comp_list = []
                        return 
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
                        if (raw_input_np[ii][0] != 0): # and (np.ptp(comp_input_np[ii, 1:]) > 2)): # and (np.sum(comp_input_np[ii, 1:] == 0) > 60):
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
                else:
                    print("Skipping for other reasons...")

            else:
                #print("Raw path exists, but compressed path doesn't: " + str(full_path))
                xy = 1
        #break

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
    #raw_arr_norm = tf.divide(tf.subtract(raw_arr, tf.reduce_min(raw_arr)), 
    #                        tf.subtract(tf.reduce_max(raw_arr), tf.reduce_min(raw_arr))
    #                    )
    raw_arr_norm = raw_arr
    comp_arr = comp_arr[None, :]
    raw_arr_norm = raw_arr_norm[None, :]
    return comp_arr, raw_arr_norm

def train(remake=False, use_chk=False, make_test=False, plot=False, show=False, save=False, epochs=1, plot_idx=0):

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
    num_train = 30
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

    num_use = 1
    #file_idx = list(range(num_use))
    # Shuffle the files to prevent grouping
    #random.shuffle(file_idx)

    file_idx = [0] #[54, 10, 32]

    final_files = []
    total_size = 0
    for i in range(num_use):
        final_files.append(f"tf_records_{file_idx[i]}.tfrecordz")
        # Accumulate the number of examples in each file
        total_size += int(size_arr[file_idx[i]])

    print(f"Files for training: {final_files}")
    print(f"TOTAL NUMBER OF EXAMPLES: {total_size}")

    newds = tf.compat.v2.data.TFRecordDataset(final_files, compression_type="GZIP", num_parallel_reads=16)
    buffer_size = 3000000
    batch_size = 256

    newds = newds.map(parse_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #newds = newds.shuffle(buffer_size)
    newds = newds.repeat()
    newds = newds.batch(batch_size) #, drop_remainder=True)
    newds = newds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #ds_it = iter(newds)

    def model_create(example_length):

        inputs = keras.Input(shape=(1, example_length, ), name="in")
        #inputs = keras.Input(shape=(example_length, ), name="in")
        #x = layers.Dense(2000, activation="relu", name="dense_1")(inputs)
        x = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='valid', activation='relu', 
                data_format='channels_first', use_bias=True, kernel_initializer='glorot_uniform')(inputs)
        x = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
        x = layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu', data_format='channels_first', 
                use_bias=True, kernel_initializer='glorot_uniform')(x)
        x = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)

        x = layers.Conv1D(filters=96, kernel_size=3, strides=1, padding='valid', activation='relu', data_format='channels_first', 
            use_bias=True, kernel_initializer='glorot_uniform')(x)
        x = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
      #  x = layers.Dense(1000, activation="relu", name="dense_1")(inputs)
      #  x = layers.BatchNormalization(name="batch_norm_1")(x)
        #x = layers.Dense(2000, activation="relu", name="dense_2")(x)
        #x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.Dense(2000, activation="relu", name="dense_4")(x)
       # x = layers.Dense(1000, activation="relu", kernel_regularizer=keras.regularizers.l2(0.002), name="dense_4")(x)
        #x = layers.BatchNormalization(name="batch_norm_2")(x)
        x = layers.Dense(2000, activation="relu", name="dense_5")(x)
        x = layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation='relu', data_format='channels_first', use_bias=True, kernel_initializer='glorot_uniform')(x)
        #x = layers.Flatten(data_format="channels_last")(x)
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


    STEPS_PER_EPOCH = math.floor(total_size / batch_size) - 1

    if training is True:
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(newds,
                   # batch_size=batch_size, # 128
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=epochs, # 100
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
            plt.savefig(f"../img/compressed_example_{pic}.png")
        plt.figure()
        plt.title('Recon...')
        plt.plot(predicted)
        if save is True:
            plt.savefig(f"../img/reconstructed_example_{pic}.png")
        plt.figure()
        plt.title('Raw...')
        plt.plot(np.reshape(example_raw, (71,1)))
        if save is True:
            plt.savefig(f"../img/actual_example_{pic}.png")
        if show is True:
            plt.show()

    #test_data = True
    test_data = False

    if make_test is True:
        if test_data is True:
            create_test_set(raw_path, comp_path, ignore_list, './test_set')

        total_file = np.load(f"test_set_0.npz")
        arr_comp = total_file['arr_0']
        spikes = arr_comp.shape[0]
        new_arr = np.copy(arr_comp)
        print(f"This many spikes...{spikes}")
        #for i in range(spikes):
        #    if(i % 1000) == 0:
        #        print(f"progress...{i}")
            #comp_re = np.reshape(arr_comp[i][2:], (1, 1, 71))
        spike_arr = arr_comp[:,2:]
        spike_arr = spike_arr[:, None ,:]
        predicted = model.predict(spike_arr, verbose=0)
        predicted = predicted.reshape((spikes, 71))
        new_arr[:, 2:] = predicted
        print("Made it...")
            # plt.figure()
            # plt.title(f'{i}_Compressed...')
            # plt.plot(np.reshape(arr_comp[i][2:], (71, 1)))
            # plt.figure()
            # plt.title(f'{i}_Recon...')
            # plt.plot(predicted)
            # plt.show()
        np.savez_compressed("final_testing_spikes.npz", new_arr)

if __name__=="__main__":
    train(
        remake=False, # Create new npz
        use_chk=True, # Use checkpointed model (don't train again)
        make_test=True,
        plot=False, # plot the figures at the end
        show=False, # show any plots
        save=False, # save the images
        epochs=10,
        plot_idx=150 # which item to plot
    ) 

