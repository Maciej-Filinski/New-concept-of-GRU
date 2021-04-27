from NewGRU import NewGRU
from tensorflow import keras
import tensorflow as tf
from SystemDefinition import LinearSystem
import matplotlib.pyplot as plt
import numpy as np


def generate_data_from_nn():
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
    inputs = tf.random.normal(shape=(batch_size, time_step, numbers_of_system_input), mean=0, stddev=2, dtype='float32')
    outputs = model(inputs)
    return inputs, outputs


def generate_data_from_linear_system(number_of_sample):
    state_matrix = np.array([[0, 1, 0], [0, 0, 1], [-1, -2, -3]])
    input_matrix = np.array([[0], [0], [1]])
    output_matrix = np.array([[1, 1, 1]])
    system = LinearSystem(state_matrix=state_matrix,
                          input_matrix=input_matrix,
                          output_matrix=output_matrix)
    initial_state = np.array([[0], [0], [0]])
    input_sequence = np.random.uniform(-1.73, 1.73, size=(number_of_sample, 1))
    output_sequence = system.linear_system_response(input_sequence=input_sequence,
                                                    initial_state=initial_state)
    return input_sequence, output_sequence


def prepare_data_for_neural_network(input_sequence, neural_network_input_length):
    numbers_of_sample = np.shape(input_sequence)[0]
    numbers_of_input = np.shape(input_sequence)[1]
    neural_network_input_sequence = np.zeros(shape=(numbers_of_sample - neural_network_input_length,
                                                    neural_network_input_length,
                                                    numbers_of_input))
    for i in range(numbers_of_sample - neural_network_input_length):
        for j in range(neural_network_input_length):
            neural_network_input_sequence[i, :, :] = input_sequence[i: i + neural_network_input_length, :]
    return neural_network_input_sequence


if __name__ == '__main__':
    input_seq, output_seq = generate_data_from_linear_system(number_of_sample=100)

    nn_input = prepare_data_for_neural_network(input_sequence=input_seq,
                                               neural_network_input_length=10)
    print(np.shape(nn_input))
    print(input_seq)
    print(nn_input)
    input()
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

    inputs, outputs = generate_data_from_nn()
    outputs = tf.reshape(outputs, [10032, 1, 2])
    before_training = model.predict(inputs[8000:10000, :, :])
    print(inputs.shape)
    print(outputs.shape)
    model.fit(inputs[0:5000, :, :], outputs[0:5000, :, :],
              epochs=2,
              validation_data=(inputs[5000:8000, :, :], outputs[5000: 8000, :, :]))
    predict_output = model.predict(inputs[8000:10000, :, :])
    tf.print(outputs[8000: 10000, 0, 0])
    tf.print(predict_output)
    plt.plot(outputs[8000: 10000, 0, 0], label='real output')
    plt.plot(predict_output[:, 0], label='predict output')
    plt.plot(before_training[:, 0], label='before_training')
    plt.legend()
    plt.figure()
    plt.plot(outputs[8000: 10000, 0, 1], label='real output')
    plt.plot(predict_output[:, 1], label='predict output')
    plt.plot(before_training[:, 1], label='before_training')
    plt.legend()
    plt.show()
