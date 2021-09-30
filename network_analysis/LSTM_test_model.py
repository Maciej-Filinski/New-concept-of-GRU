import tensorflow as tf
from tensorflow import keras
from data_generator import ToyProblemRealRoots, ToyProblemComplexRoots, ToyProblemOriginal
import numpy as np
import matplotlib.pyplot as plt
from layers import BasicLSTM

EPOCHS = 500
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 100

STATE_LENGTH = 16
data_generator = ToyProblemComplexRoots(number_of_train_samples=TRAIN_SAMPLES, number_of_test_samples=TEST_SAMPLES,
                                        dataset_name='toy_problem_complex_roots')

dataset = data_generator.load_data()


build_in_model = keras.Sequential()
build_in_model.add(keras.layers.LSTM(STATE_LENGTH, batch_input_shape=(1, 1, 1), stateful=True))
build_in_model.add(keras.layers.Dense(1))
build_in_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                       metrics=[tf.keras.metrics.RootMeanSquaredError()])
build_in_model.summary()
initial_weight = build_in_model.get_weights()

cell = BasicLSTM(state_length=STATE_LENGTH)
rnn = keras.layers.RNN(cell, stateful=True)
input_layer = keras.Input(batch_shape=(1, None, 1), ragged=True)
output = rnn(input_layer)
output = keras.layers.Dense(1)(output)
custom_model = keras.models.Model(input_layer, output)
custom_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[tf.keras.metrics.RootMeanSquaredError()])
custom_model.summary()
possible = tf.split(initial_weight[1], 4, axis=1)

custom_model.set_weights(initial_weight)

custom_model.reset_states()
build_in_model.reset_states()

history_build_in = build_in_model.fit(dataset['train']['inputs'], dataset['train']['outputs'], batch_size=1,
                                      epochs=EPOCHS, shuffle=False,
                                      validation_data=(dataset['test']['inputs'], dataset['test']['outputs']))

history_custom = custom_model.fit(dataset['train']['inputs'], dataset['train']['outputs'], batch_size=1,
                                  epochs=EPOCHS, shuffle=False,
                                  validation_data=(dataset['test']['inputs'], dataset['test']['outputs']))

build_in_model.reset_states()
predicted_outputs_build_in = np.zeros((TEST_SAMPLES, 1))
for i in range(TEST_SAMPLES):
    predicted_outputs_build_in[i, :] = build_in_model.predict(dataset['test']['inputs'][i: i + 1, :, :])

custom_model.reset_states()
predicted_outputs_custom = np.zeros((TEST_SAMPLES, 1))
for i in range(TEST_SAMPLES):
    predicted_outputs_custom[i, :] = custom_model.predict(dataset['test']['inputs'][i: i + 1, :, :])


fig, axs = plt.subplots(3, 2)
fig.suptitle(f'{TRAIN_SAMPLES=}, {TEST_SAMPLES=}, {EPOCHS=}, LSTM_{STATE_LENGTH=}')
axs[0, 0].plot(dataset['test']['outputs'], label='true outputs')
axs[0, 0].plot(predicted_outputs_build_in, label='predicted outputs')
axs[0, 0].legend()
axs[0, 0].grid()
axs[0, 0].set_title('build-in LSTM')
axs[1, 0].plot(dataset['test']['outputs'], label='true outputs')
axs[1, 0].plot(predicted_outputs_custom, label='predicted outputs')
axs[1, 0].legend()
axs[1, 0].grid()
axs[1, 0].set_title('custom LSTM')
axs[2, 0].plot(predicted_outputs_build_in, label='build-in')
axs[2, 0].plot(predicted_outputs_custom, label='custom')
axs[2, 0].plot(abs(predicted_outputs_build_in - predicted_outputs_custom), label='difference')
axs[2, 0].legend()
axs[2, 0].grid()
axs[2, 0].set_title(f'build-in LSTM vs custom LSTM')

axs[0, 1].plot(history_build_in.history['loss'], label='train loss')
axs[0, 1].plot(history_build_in.history['val_loss'], label='test loss')
axs[0, 1].legend()
axs[0, 1].grid()
axs[0, 1].set_title('build-in LSTM')
axs[1, 1].plot(history_custom.history['loss'], label='train loss')
axs[1, 1].plot(history_custom.history['val_loss'], label='test loss')
axs[1, 1].legend()
axs[1, 1].grid()
axs[1, 1].set_title('custom LSTM')
axs[2, 1].plot(history_custom.history['loss'], label='build-in LSTM train loss')
axs[2, 1].plot(history_build_in.history['loss'], label='custom LSTM train loss')
axs[2, 1].plot(abs(np.array(history_custom.history['loss']) - np.array(history_build_in.history['loss'])), label='difference')
axs[2, 1].set_xlabel('epoch')
axs[2, 1].legend()
axs[2, 1].grid()
axs[2, 1].set_title(f'build-in LSTM vs custom LSTM')
plt.show()
