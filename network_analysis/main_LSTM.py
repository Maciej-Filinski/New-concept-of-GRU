import tensorflow as tf
from tensorflow import keras
from data_generator import ToyProblemRealRoots, ToyProblemComplexRoots, ToyProblemOriginal
import numpy as np
import matplotlib.pyplot as plt
from layers import BasicLSTM

EPOCHS = 100

neural_network = keras.Sequential()
neural_network.add(keras.layers.LSTM(8, batch_input_shape=(1, 1, 1), stateful=True))
neural_network.add(keras.layers.Dense(1))
neural_network.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                       metrics=[tf.keras.metrics.RootMeanSquaredError()])
neural_network.summary()

data_generator = ToyProblemComplexRoots(number_of_train_samples=500, number_of_test_samples=500,
                                        dataset_name='toy_problem_complex_roots')

dataset = data_generator.load_data()

neural_network.fit(dataset['train']['inputs'], dataset['train']['outputs'], batch_size=1, epochs=EPOCHS, shuffle=False,
                   validation_data=(dataset['test']['inputs'], dataset['test']['outputs']))

neural_network.reset_states()
predicted_outputs = np.zeros((500, 1))
for i in range(500):
    predicted_outputs[i, :] = neural_network.predict(dataset['test']['inputs'][i: i + 1, :, :])

plt.plot(dataset['test']['outputs'], label='true outputs')
plt.plot(predicted_outputs, label='predicted outputs')
plt.legend()

cell = BasicLSTM(state_length=8)
rnn = keras.layers.RNN(cell, stateful=True)
input_layer = keras.Input(batch_shape=(1, None, 1), ragged=True)
output = rnn(input_layer)
output = keras.layers.Dense(1)(output)
model = keras.models.Model(input_layer, output)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()

neural_network.fit(dataset['train']['inputs'], dataset['train']['outputs'], batch_size=1, epochs=EPOCHS, shuffle=False,
                   validation_data=(dataset['test']['inputs'], dataset['test']['outputs']))

neural_network.reset_states()
predicted_outputs = np.zeros((500, 1))
for i in range(500):
    predicted_outputs[i, :] = neural_network.predict(dataset['test']['inputs'][i: i + 1, :, :])

plt.plot(dataset['test']['outputs'], label='true outputs')
plt.plot(predicted_outputs, label='predicted outputs')
plt.legend()
plt.show()
