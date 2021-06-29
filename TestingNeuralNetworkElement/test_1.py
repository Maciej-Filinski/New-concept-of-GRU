from NeuralNetwork import NewGRU
import tensorflow as tf
from tensorflow import keras
import constants as c
import numpy as np
import matplotlib.pyplot as plt

STATE_SPACE_VECTOR_LENGTH = 2
NUMBER_OF_INPUTS = 1

cell = NewGRU(state_length=c.NEURAL_NETWORK_STATE_LENGTH,
              number_of_outputs=STATE_SPACE_VECTOR_LENGTH,
              input_dnn_structure=c.INPUT_DNN_STRUCTURE,
              forget_dnn_structure=c.FORGET_DNN_STRUCTURE,
              output_dnn_structure=c.OUTPUT_DNN_STRUCTURE,
              forget_dnn_enable=False,
              output_dnn_enable=False)
rnn = keras.layers.RNN(cell, stateful=True)
input_layer = keras.Input(batch_shape=(1, None, NUMBER_OF_INPUTS))
output = rnn(input_layer)
model = keras.models.Model(input_layer, output)
model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.metrics.RootMeanSquaredError())
model.summary()

number_of_sample_train = 10000
number_of_sample_test = 100
train_inputs = np.random.uniform(-1, 1, size=(number_of_sample_train, 1, NUMBER_OF_INPUTS))
train_outputs = np.zeros(shape=(number_of_sample_train + 1, STATE_SPACE_VECTOR_LENGTH))
test_inputs = np.sin(2 * np.pi * np.array(range(number_of_sample_test)) / 10)
test_inputs = np.zeros(shape=(number_of_sample_test, NUMBER_OF_INPUTS))
test_inputs[0, 0] = 1
test_inputs = np.reshape(test_inputs, newshape=(number_of_sample_test, 1, NUMBER_OF_INPUTS))
test_outputs = np.zeros(shape=(number_of_sample_test + 1, STATE_SPACE_VECTOR_LENGTH))

state_matrix = np.array([[0.7, 0.8], [0, 0.1]])
input_matrix = np.array([[-1], [0.1]])

for i in range(1, number_of_sample_train + 1):
    train_outputs[i, :] = state_matrix @ train_outputs[i - 1, :] + input_matrix @ train_inputs[i - 1, 0, :]
train_outputs = train_outputs[1::, :]

for i in range(1, number_of_sample_test + 1):
    test_outputs[i, :] = state_matrix @ test_outputs[i - 1, :] + input_matrix @ test_inputs[i - 1, 0, :]
test_outputs = test_outputs[1::, :]
"""
weight = model.get_weights()
weight[1] = np.array([0, 0])
weight[0] = np.array([[-1, 0.1], [0.7, 0], [0.8, 0.1]])
model.set_weights(weight)"""
mse = tf.keras.losses.MeanSquaredError()
predicted_train_outputs = np.zeros(shape=(number_of_sample_test, STATE_SPACE_VECTOR_LENGTH))
for i in range(number_of_sample_test):
    predicted_train_outputs[i, :] = model.predict(train_inputs[i, :, :])
print(mse(train_outputs[:100, :], predicted_train_outputs))

model.fit(train_inputs, train_outputs, epochs=10, batch_size=1, validation_data=(test_inputs, test_outputs))


model.reset_states()
predicted_train_outputs = np.zeros(shape=(number_of_sample_test, STATE_SPACE_VECTOR_LENGTH))
for i in range(number_of_sample_test):
    predicted_train_outputs[i, :] = model.predict(train_inputs[i, :, :])

mse = tf.keras.losses.MeanSquaredError()
print(mse(train_outputs[:100, :], predicted_train_outputs))
model.reset_states()
predicted_test_outputs = np.zeros(shape=(number_of_sample_test, STATE_SPACE_VECTOR_LENGTH))
for i in range(number_of_sample_test):
    predicted_test_outputs[i, :] = model.predict(test_inputs[i, :, :])

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

