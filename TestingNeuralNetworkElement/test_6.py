import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data_generator import *
from build_model import *

"""
Simulation parameters
WARNING!
TRAIN_DATA_LENGTH must be a multiple of the BATCH_SIZE and higher or equal TEST_DATA_LENGTH
"""
EPOCHS = 20
BATCH_SIZE = 50
TRAIN_DATA_LENGTH = 1000
TEST_DATA_LENGTH = 100
NUMBER_OF_INPUTS = 1
NUMBER_OF_OUTPUTS = 1
NOISE = False

"""
Learning model
"""
data = data_nonlinear_system_input_output(number_of_sample_train=TRAIN_DATA_LENGTH,
                                          number_of_sample_test=TEST_DATA_LENGTH)
x_train_dataset, y_train_dataset, x_test_dataset, y_test_dataset, state = data

model = build_model_input_output(number_of_outputs=NUMBER_OF_INPUTS,
                                 number_of_inputs=NUMBER_OF_OUTPUTS,
                                 batch_size=BATCH_SIZE,
                                 struct=1,
                                 stateful=False)
model.summary()
x_train = tf.ragged.constant([x_train_dataset[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
y_train = np.copy(y_train_dataset[0:TRAIN_DATA_LENGTH, :])
if NOISE is True:
    y_train += np.random.normal(0, 0.5, size=(TRAIN_DATA_LENGTH, 1))
y_train = y_train / 10
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
predicted_train_outputs = 10 * predicted_train_outputs
model.reset_states()
predicted_test_outputs = np.zeros(shape=(TEST_DATA_LENGTH, NUMBER_OF_OUTPUTS))
neural_network_test_state = np.zeros(shape=(TEST_DATA_LENGTH, model_state_length))
neural_network_test_candidate = np.zeros(shape=(TEST_DATA_LENGTH, model_state_length))
neural_network_test_forget = np.zeros(shape=(TEST_DATA_LENGTH, model_state_length))
for j in range(TEST_DATA_LENGTH):
    predicted_test_outputs[j, :], neural_network_test_state[j, :], neural_network_test_forget[j, :], neural_network_test_candidate[j, :] = model.predict(x_test_dataset[j, :, :])
predicted_test_outputs = 10 * predicted_test_outputs
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

axs[1, 0].plot(state[0][:TEST_DATA_LENGTH, :], 'g', label='true state', linewidth=2)
axs[1, 0].plot(neural_network_train_state[:TEST_DATA_LENGTH, :], '--', label='predicted state', linewidth=2)
axs[1, 0].legend()
axs[1, 0].set_ylabel(r'$h_n$')
axs[2, 0].plot(neural_network_train_candidate[:TEST_DATA_LENGTH, :], linewidth=2)
axs[2, 0].legend(['variable ' + str(i + 1) for i in range(model_state_length)])
axs[2, 0].set_ylabel(r'$\hat{h}_n$')
axs[3, 0].plot(neural_network_train_forget[:TEST_DATA_LENGTH, :], linewidth=2)
axs[3, 0].legend(['variable ' + str(i + 1) for i in range(model_state_length)])
axs[3, 0].set_ylabel(r'$f_n$')

axs[1, 1].plot(state[1][:TEST_DATA_LENGTH, :], 'g', label='true state', linewidth=2)
axs[1, 1].plot(neural_network_test_state[:TEST_DATA_LENGTH, :], '--', label='predicted state', linewidth=2)
axs[1, 1].legend()
axs[2, 1].plot(neural_network_test_candidate[:TEST_DATA_LENGTH, :], linewidth=2)
axs[2, 1].legend(['variable ' + str(i + 1) for i in range(model_state_length)])
axs[3, 1].plot(neural_network_test_forget[:TEST_DATA_LENGTH, :], linewidth=2)
axs[3, 1].legend(['variable ' + str(i + 1) for i in range(model_state_length)])
axs[3, 0].set_xlabel(r'$n$')
axs[3, 1].set_xlabel(r'$n$')
plt.show()
