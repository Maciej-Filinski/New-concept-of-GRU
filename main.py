from NewGRU import NewGRU
from tensorflow import keras
from tensorflow.keras.layers import RNN
import tensorflow as tf
from SystemDefinition import LinearSystem
import matplotlib.pyplot as plt
import numpy as np



def run_test():
    state_length = 16
    time_step = 32
    numbers_of_system_input = 10
    numbers_of_system_output = 2
    batch_size = 1

    cell = NewGRU(state_length, numbers_of_system_output)
    rnn = keras.layers.RNN(cell)
    input_1 = keras.Input((time_step, numbers_of_system_input))
    outputs = rnn(input_1)
    model = keras.models.Model(input_1, outputs)

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.summary()
    test_output = model(tf.random.uniform(shape=(batch_size, time_step, numbers_of_system_input)))
    print(f'Neural network running test. Output = {test_output}')


if __name__ == '__main__':
    run_test()
    state_length = 100
    time_step = 32
    numbers_of_system_input = 8
    numbers_of_system_output = 2
    batch_size = 10032

    cell = NewGRU(state_length, numbers_of_system_output)
    rnn = keras.layers.RNN(cell)
    input_1 = keras.Input((time_step, numbers_of_system_input))
    outputs = rnn(input_1)
    model = keras.models.Model(input_1, outputs)

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.summary()

    inputs = tf.random.uniform(shape=(batch_size, 1, numbers_of_system_input), dtype='float32')
    outputs = tf.add(inputs[:, :, 0: 2], inputs[:, :, 3: 5])
    outputs = tf.add(outputs, inputs[:, :, 6: 8])
    tmp = inputs[0: batch_size - time_step, :, :]
    for i in range(1, time_step):
        tmp = tf.concat((tmp, inputs[i: batch_size - time_step + i, :, :]), axis=1)
    inputs = tmp
    model.fit(inputs[0:5000, :, :], outputs[0:5000, :, :],
              epochs=10,
              validation_data=(inputs[5000:8000, :, :], outputs[5000: 8000, :, :]))
    predict_output = model.predict(inputs[8000:10000, :, :])
    plt.plot(outputs[8000: 10000, 0, 0], label='real output')
    plt.plot(predict_output[:, 0], label='predict output')
    plt.legend()
    plt.figure()
    plt.plot(outputs[8000: 10000, 0, 1], label='real output')
    plt.plot(predict_output[:, 1], label='predict output')
    plt.legend()
    plt.show()
