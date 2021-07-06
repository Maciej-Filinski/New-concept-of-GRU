"""
Test: only candidate neural network.

Learning test for non-linear state space system initial state = 0.
Eq 11. from D. Masti, A. Bemporad - Learning nonlinear state-space models using autoencoders.

Neural network input => system input sequence.
Neural network output => state space vector.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from build_model import build_model

STATE_SPACE_VECTOR_LENGTH = 2
NUMBER_OF_INPUTS = 1

"""
Prepare data from system.
"""
number_of_sample_train = 2000
number_of_sample_test = 100
train_inputs = np.random.uniform(-1, 1, size=(number_of_sample_train, 1, NUMBER_OF_INPUTS))
train_outputs = np.zeros(shape=(number_of_sample_train + 1, STATE_SPACE_VECTOR_LENGTH))
test_inputs = np.sin(2 * np.pi * np.array(range(number_of_sample_test)) / 50)
test_inputs = np.reshape(test_inputs, newshape=(number_of_sample_test, 1, NUMBER_OF_INPUTS))
test_outputs = np.zeros(shape=(number_of_sample_test + 1, STATE_SPACE_VECTOR_LENGTH))

state_matrix = np.array([[0.7555, 0.25], [-0.1991, 0]])
input_matrix = np.array([[-0.5], [0]])

for i in range(1, number_of_sample_train + 1):
    if train_inputs[i - 1, 0, :] > 0:
        train_outputs[i, :] = state_matrix @ train_outputs[i - 1, :] + input_matrix @ np.sqrt(train_inputs[i - 1, 0, :])
    else:
        train_outputs[i, :] = state_matrix @ train_outputs[i - 1, :] + input_matrix @ train_inputs[i - 1, 0, :]

train_outputs = train_outputs[1::, :]
for i in range(1, number_of_sample_test + 1):
    if test_inputs[i - 1, 0, :] > 0:
        test_outputs[i, :] = state_matrix @ test_outputs[i - 1, :] + input_matrix @ np.sqrt(test_inputs[i - 1, 0, :])
    else:
        test_outputs[i, :] = state_matrix @ test_outputs[i - 1, :] + input_matrix @ test_inputs[i - 1, 0, :]
test_outputs = test_outputs[1::, :]

train_data_length = 2000
x_train = tf.ragged.constant([train_inputs[0:i, 0, :] for i in range(1, train_data_length + 1)])
y_train = train_outputs[0:train_data_length, :]


""" 
Build and learn model.
"""
batch_size = 10
model = build_model(ss_vector_length=STATE_SPACE_VECTOR_LENGTH,
                    number_of_inputs=NUMBER_OF_INPUTS,
                    batch_size=batch_size)
model.summary()
model.fit(x_train, y_train, epochs=50, batch_size=batch_size)
weights = model.get_weights()
model.save_weights('./models/test_3/structure_1/')

"""
Change batch size and prepare for plot result.
"""
model = build_model(ss_vector_length=STATE_SPACE_VECTOR_LENGTH,
                    number_of_inputs=NUMBER_OF_INPUTS,
                    batch_size=1)
model.set_weights(weights)
model.reset_states()
predicted_train_outputs = np.zeros(shape=(number_of_sample_test, STATE_SPACE_VECTOR_LENGTH))
for i in range(number_of_sample_test):
    predicted_train_outputs[i, :] = model.predict(train_inputs[i, :, :])

model.reset_states()
predicted_test_outputs = np.zeros(shape=(number_of_sample_test, STATE_SPACE_VECTOR_LENGTH))
for i in range(number_of_sample_test):
    predicted_test_outputs[i, :] = model.predict(test_inputs[i, :, :])

"""
Plot result.
"""
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(train_outputs[:number_of_sample_test, 0], 'b', label='true state', linewidth=4)
axs[0, 0].plot(predicted_train_outputs[:, 0], '--r', label='predicted state', linewidth=2)
axs[0, 0].set_title('train: x_1')
axs[0, 0].legend()
axs[0, 1].plot(train_outputs[:number_of_sample_test, 1], 'b', label='true state', linewidth=4)
axs[0, 1].plot(predicted_train_outputs[:, 1], '--r', label='predicted state', linewidth=2)
axs[0, 1].set_title('train: x_2')
axs[0, 1].legend()
axs[1, 0].plot(test_outputs[:, 0], 'b', label='true state', linewidth=4)
axs[1, 0].plot(predicted_test_outputs[:, 0], '--r', label='predicted state', linewidth=2)
axs[1, 0].set_title('test: x_1 - impulse response')
axs[1, 0].legend()
axs[1, 1].plot(test_outputs[:, 1], 'b', label='true state', linewidth=4)
axs[1, 1].plot(predicted_test_outputs[:, 1], '--r', label='predicted state', linewidth=2)
axs[1, 1].set_title('test: x_2 - impulse response')
axs[1, 1].legend()
plt.show()
