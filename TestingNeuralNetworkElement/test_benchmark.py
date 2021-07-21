import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data_generator import *
from build_model import *
from pathlib import Path
import scipy.io
import os
import urllib
import urllib.request
import zipfile

"""
Simulation parameters
WARNING!
TRAIN_DATA_LENGTH must be a multiple of the BATCH_SIZE and higher or equal TEST_DATA_LENGTH
"""
EPOCHS = 2
BATCH_SIZE = 40
TRAIN_DATA_LENGTH = 81920
TEST_DATA_LENGTH = 40400
NUMBER_OF_INPUTS = 1
NUMBER_OF_OUTPUTS = 1


def benchmark_data():
    """
    Load Silverbox data.
    Note: Based on https://github.com/antonior92/sysid-neuralnet/blob/master/python-scripts/data/silverbox.py
    """
    # Extract input and output data Silverbox
    mat = scipy.io.loadmat(maybe_download_and_extract())
    u = mat['V1'][0]  # Input
    y = mat['V2'][0]  # Output

    # Number of samples of each subset of data
    n_zeros = 100  # Number of zeros at the start
    n_test = 40400  # Number of samples in the test set
    n_trans_before = 460  # Number of transient samples before each multisine realization
    n = 8192  # Number of samples per multisine realization
    n_trans_after = 40  # Number of transient samples after each multisine realization
    n_block = n_trans_before + n + n_trans_after
    n_multisine = 10  # Number of multisine realizations
    
    # Extract training data
    u_train = np.zeros(n_multisine * n)
    y_train = np.zeros(n_multisine * n)
    for i, r in enumerate(n_multisine):
        u_train[i * n + np.arange(n)] = u[n_zeros + n_test + r * n_block + n_trans_before + np.arange(n)]
        y_train[i * n + np.arange(n)] = y[n_zeros + n_test + r * n_block + n_trans_before + np.arange(n)]
    
    # Extract test data
    u_test = u[n_zeros:n_zeros + n_test]
    y_test = y[n_zeros:n_zeros + n_test]

    # Reshape to correct dimensions
    x_train_benchmark = np.reshape(u_train, newshape=(TRAIN_DATA_LENGTH, NUMBER_OF_INPUTS))
    y_train_benchmark = np.reshape(y_train, newshape=(TRAIN_DATA_LENGTH, NUMBER_OF_OUTPUTS))
    x_test_benchmark = np.reshape(u_test, newshape=(TEST_DATA_LENGTH, NUMBER_OF_INPUTS))
    y_test_benchmark = np.reshape(y_test, newshape=(TEST_DATA_LENGTH, NUMBER_OF_OUTPUTS))

    return x_train_benchmark, y_train_benchmark, x_test_benchmark, y_test_benchmark

def maybe_download_and_extract():
    """
    Download the data from nonlinear benchmark website, unless it's already here.
    Note: Taken from https://github.com/antonior92/sysid-neuralnet/blob/master/python-scripts/data/silverbox.py
    """
    src_url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/SILVERBOX/SilverboxFiles.zip'
    home = Path.home()
    work_dir = str(home.joinpath('datasets/SilverBox'))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    zipfilepath = os.path.join(work_dir, "SilverboxFiles.zip")
    if not os.path.exists(zipfilepath):
        filepath, _ = urllib.request.urlretrieve(
            src_url, zipfilepath)
        file = os.stat(filepath)
        size = file.st_size
        print('Successfully downloaded', 'SilverboxFiles.zip', size, 'bytes.')
    else:
        print('SilverboxFiles.zip', 'already downloaded!')

    datafilepath = os.path.join(work_dir, "SilverboxFiles/SNLS80mV.mat")
    print(datafilepath)
    if not os.path.exists(datafilepath):
        zip_ref = zipfile.ZipFile(zipfilepath, 'r')
        zip_ref.extractall(work_dir)
        zip_ref.close()
        print('Successfully unzipped data')
    return datafilepath

"""
Learning model
"""
data = benchmark_data()
x_train_dataset, y_train_dataset, x_test_dataset, y_test_dataset = data

model = build_model_input_output(number_of_outputs=NUMBER_OF_INPUTS,
                                 number_of_inputs=NUMBER_OF_OUTPUTS,
                                 batch_size=BATCH_SIZE,
                                 struct=1,
                                 stateful=False)
model.summary()
x_train = tf.ragged.constant([x_train_dataset[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
y_train = np.copy(y_train_dataset[0:TRAIN_DATA_LENGTH, :])
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
weights = model.get_weights()

""" 
Testing model
"""
model = build_model_input_output(number_of_outputs=NUMBER_OF_INPUTS,
                                 number_of_inputs=NUMBER_OF_OUTPUTS,
                                 batch_size=1,
                                 struct=1,
                                 full_output=True)
model.set_weights(weights)
model.summary()
model_state_length = model.output_shape[1][1]
model.reset_states()
predicted_train_outputs = np.zeros(shape=(TEST_DATA_LENGTH, NUMBER_OF_OUTPUTS))
neural_network_train_state = np.zeros(shape=(TEST_DATA_LENGTH, model_state_length))
neural_network_train_candidate = np.zeros(shape=(TEST_DATA_LENGTH, model_state_length))
neural_network_train_forget = np.zeros(shape=(TEST_DATA_LENGTH, model_state_length))
for j in range(TEST_DATA_LENGTH):
    predicted_train_outputs[j, :], neural_network_train_state[j, :], neural_network_train_forget[j, :], neural_network_train_candidate[j, :] = model.predict(x_train_dataset[j, :, :])
predicted_train_outputs = predicted_train_outputs
model.reset_states()
predicted_test_outputs = np.zeros(shape=(TEST_DATA_LENGTH, NUMBER_OF_OUTPUTS))
neural_network_test_state = np.zeros(shape=(TEST_DATA_LENGTH, model_state_length))
neural_network_test_candidate = np.zeros(shape=(TEST_DATA_LENGTH, model_state_length))
neural_network_test_forget = np.zeros(shape=(TEST_DATA_LENGTH, model_state_length))
for j in range(TEST_DATA_LENGTH):
    predicted_test_outputs[j, :], neural_network_test_state[j, :], neural_network_test_forget[j, :], neural_network_test_candidate[j, :] = model.predict(x_test_dataset[j, :, :])
predicted_test_outputs = predicted_test_outputs
"""
Plot result
"""
fig, axs = plt.subplots(4, 2)
axs[0, 0].plot(y_train_dataset[:TEST_DATA_LENGTH, :], 'g', label='true output', linewidth=2)
axs[0, 0].plot(predicted_train_outputs[:, :], '--r', label='predicted output', linewidth=2)
axs[0, 0].set_title('TRAIN')
axs[0, 0].legend()
axs[0, 0].set_ylabel(r'$\hat{y}_n$')

axs[0, 1].plot(y_test_dataset[:TEST_DATA_LENGTH, :], 'g', label='true output', linewidth=2)
axs[0, 1].plot(predicted_test_outputs[:, :], '--r', label='predicted output', linewidth=2)
axs[0, 1].set_title('TEST')
axs[0, 1].legend()

axs[1, 0].plot(neural_network_train_state[:TEST_DATA_LENGTH, :], '--', label='predicted state', linewidth=2)
axs[1, 0].legend()
axs[1, 0].set_ylabel(r'$h_n$')
axs[2, 0].plot(neural_network_train_candidate[:TEST_DATA_LENGTH, :], linewidth=2)
axs[2, 0].legend(['variable ' + str(i + 1) for i in range(model_state_length)])
axs[2, 0].set_ylabel(r'$\hat{h}_n$')
axs[3, 0].plot(neural_network_train_forget[:TEST_DATA_LENGTH, :], linewidth=2)
axs[3, 0].legend(['variable ' + str(i + 1) for i in range(model_state_length)])
axs[3, 0].set_ylabel(r'$f_n$')

axs[1, 1].plot(neural_network_test_state[:TEST_DATA_LENGTH, :], '--', label='predicted state', linewidth=2)
axs[1, 1].legend()
axs[2, 1].plot(neural_network_test_candidate[:TEST_DATA_LENGTH, :], linewidth=2)
axs[2, 1].legend(['variable ' + str(i + 1) for i in range(model_state_length)])
axs[3, 1].plot(neural_network_test_forget[:TEST_DATA_LENGTH, :], linewidth=2)
axs[3, 1].legend(['variable ' + str(i + 1) for i in range(model_state_length)])
axs[3, 0].set_xlabel(r'$n$')
axs[3, 1].set_xlabel(r'$n$')
plt.show()
