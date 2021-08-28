# import tensorflow as tf
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from data_generator import *
# import datetime
# import scipy.io
# import os
# import urllib
# import urllib.request
# import zipfile
# import time
# from pathlib import Path
# import random
#
#
class Silverbox:
    pass
#
# """
# simulation parameters
# WARNING!
# TRAIN_DATA_LENGTH must be a multiple of the BATCH_SIZE and higher or equal TEST_DATA_LENGTH
# """
# EPOCHS = 100
# BATCH_SIZE = 10
# TRAIN_DATA_LENGTH = 81920
# TEST_DATA_LENGTH = 40400
# NUMBER_OF_INPUTS = 1
# NUMBER_OF_OUTPUTS = 1
# NOISE = False
#
# cell = NewGRU(state_length=s2.NEURAL_NETWORK_STATE_LENGTH,
#               number_of_outputs=NUMBER_OF_OUTPUTS,
#               candidate_dnn_structure=s2.CANDIDATE_DNN_STRUCTURE,
#               forget_dnn_structure=s2.FORGET_DNN_STRUCTURE,
#               output_dnn_structure=s2.OUTPUT_DNN_STRUCTURE,
#               forget_dnn_enable=True,
#               output_dnn_enable=True,
#               full_output=True)
# rnn = keras.layers.RNN(cell, stateful=True)
# input_layer = keras.Input(batch_shape=(1, None, NUMBER_OF_INPUTS), ragged=True)
# output = rnn(input_layer)
# model = tf.keras.Model(input_layer, output)
# model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
#               metrics=[tf.keras.metrics.RootMeanSquaredError()])
# def benchmark_data():
#     """
#     Load Silverbox datasets.
#     Note: Based on https://github.com/antonior92/sysid-neuralnet/blob/master/python-scripts/data/silverbox.py
#     """
#     # Extract input and output datasets Silverbox
#     mat = scipy.io.loadmat(maybe_download_and_extract())
#     u = mat['V1'][0]  # Input
#     y = mat['V2'][0]  # Output
#
#     # Number of samples of each subset of datasets
#     n_zeros = 100  # Number of zeros at the start
#     n_test = 40400  # Number of samples in the test set
#     n_trans_before = 460  # Number of transient samples before each multisine realization
#     n = 8192  # Number of samples per multisine realization
#     n_trans_after = 40  # Number of transient samples after each multisine realization
#     n_block = n_trans_before + n + n_trans_after
#     n_multisine = 10  # Number of multisine realizations
#
#     # Extract training datasets
#     u_train = np.zeros(n_multisine * n)
#     y_train = np.zeros(n_multisine * n)
#     for i, r in enumerate(range(n_multisine)):
#         u_train[i * n + np.arange(n)] = u[n_zeros + n_test + r * n_block + n_trans_before + np.arange(n)]
#         y_train[i * n + np.arange(n)] = y[n_zeros + n_test + r * n_block + n_trans_before + np.arange(n)]
#
#     # Extract test datasets
#     u_test = u[n_zeros:n_zeros + n_test]
#     y_test = y[n_zeros:n_zeros + n_test]
#
#     # Reshape to correct dimensions
#     x_train_benchmark = np.reshape(u_train, newshape=(TRAIN_DATA_LENGTH, NUMBER_OF_INPUTS))
#     y_train_benchmark = np.reshape(y_train, newshape=(TRAIN_DATA_LENGTH, NUMBER_OF_OUTPUTS))
#     x_test_benchmark = np.reshape(u_test, newshape=(TEST_DATA_LENGTH, NUMBER_OF_INPUTS))
#     y_test_benchmark = np.reshape(y_test, newshape=(TEST_DATA_LENGTH, NUMBER_OF_OUTPUTS))
#
#     return x_train_benchmark, y_train_benchmark, x_test_benchmark, y_test_benchmark
#
# def maybe_download_and_extract():
#     """
#     Download the datasets from nonlinear benchmark website, unless it's already here.
#     Note: Taken from https://github.com/antonior92/sysid-neuralnet/blob/master/python-scripts/data/silverbox.py
#     """
#     src_url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/SILVERBOX/SilverboxFiles.zip'
#     home = Path.home()
#     work_dir = str(home.joinpath('datasets/SilverBox'))
#     if not os.path.exists(work_dir):
#         os.makedirs(work_dir)
#     zipfilepath = os.path.join(work_dir, "SilverboxFiles.zip")
#     if not os.path.exists(zipfilepath):
#         filepath, _ = urllib.request.urlretrieve(
#             src_url, zipfilepath)
#         file = os.stat(filepath)
#         size = file.st_size
#         print('Successfully downloaded', 'SilverboxFiles.zip', size, 'bytes.')
#     else:
#         print('SilverboxFiles.zip', 'already downloaded!')
#
#     datafilepath = os.path.join(work_dir, "SilverboxFiles/SNLS80mV.mat")
#     print(datafilepath)
#     if not os.path.exists(datafilepath):
#         zip_ref = zipfile.ZipFile(zipfilepath, 'r')
#         zip_ref.extractall(work_dir)
#         zip_ref.close()
#         print('Successfully unzipped datasets')
#     return datafilepath
#