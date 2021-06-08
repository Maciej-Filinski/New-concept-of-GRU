from NeuralNetwork import NewGRU, SimpleNewGRU
from tensorflow import keras
import tensorflow as tf
from SystemDefinition import LinearSystem, NonlinearSystemBenchmark
import matplotlib.pyplot as plt
import numpy as np
from DataPreprocessing.TeacherLS import TeacherLS
from DataVisualization import print_output


def create_neural_network(number_of_inputs,
                          number_of_outputs,
                          time_step,
                          neural_network_state_length):
    cell = NewGRU(state_length=neural_network_state_length, number_of_output=number_of_outputs)
    rnn = keras.layers.RNN(cell)
    input_1 = keras.Input((time_step, number_of_inputs))
    output = rnn(input_1)
    model = keras.models.Model(input_1, output)
    model.compile(optimizer="adam", loss="mse", metrics=tf.keras.metrics.RootMeanSquaredError())
    model.summary()
    return model


if __name__ == '__main__':
    train_data_length = 100
    valid_data_length = 100
    test_data_length = 400
    nn_state_length = 20
    number_of_all_input_batch = train_data_length + valid_data_length
    time_step = 20
    number_of_system_sample = number_of_all_input_batch + time_step
    number_of_system_inputs = 1
    number_of_system_outputs = 1

    system = NonlinearSystemBenchmark()
    input_sequence = 5 * (np.random.rand(1, number_of_system_sample) - 0.5)
    output_sequence = system.response(input_sequence).transpose()
    input_sequence = input_sequence.transpose()
    nn_inputs = np.zeros(shape=(number_of_all_input_batch, time_step, 1))
    for i in range(number_of_all_input_batch):
        nn_inputs[i, :, :] = input_sequence[i: i + time_step, :]
    train_inputs = nn_inputs[0:train_data_length, :, :]
    train_outputs = output_sequence[time_step:train_data_length + time_step, :]
    val_inputs = nn_inputs[0:train_data_length, :, :]
    val_outputs = output_sequence[time_step:train_data_length + time_step, :]

    input_sequence = np.array(list(range(test_data_length + time_step))).reshape((1, test_data_length + time_step))
    input_sequence = np.sin(2 * np.pi * input_sequence / 10) + np.sin(2 * np.pi * input_sequence / 25)
    output_sequence = system.response(input_sequence).transpose()

    test_output = output_sequence[time_step::, :]
    input_sequence = input_sequence.transpose()
    test_inputs = np.zeros(shape=(test_data_length, time_step, 1))

    plt.figure(num='out')
    plt.plot(output_sequence[1:, :], label='true noisefree output', linewidth=4)
    plt.plot(input_sequence, label='true  output', linewidth=1)

    for i in range(test_data_length):
        test_inputs[i, :, :] = input_sequence[i: i + time_step, :]

    neural_network = create_neural_network(number_of_inputs=number_of_system_inputs,
                                           number_of_outputs=number_of_system_outputs,
                                           time_step=time_step,
                                           neural_network_state_length=nn_state_length)

    neural_network.fit(train_inputs,
                       train_outputs,
                       epochs=1,
                       validation_data=(val_inputs, val_outputs))
    predicted_output_train_data = neural_network(train_inputs)

    plt.figure(num='Predicted output - train test_data')
    plt.plot(train_outputs, label='true noisy output')
    plt.plot(predicted_output_train_data, label='predicted output')
    plt.xlabel('n')
    plt.ylabel('y_n')
    plt.legend()
    plt.grid()

    predicted_output = neural_network.predict(test_inputs)
    plt.figure(num='Predicted output - test test_data')
    plt.plot(test_output, label='true noisefree output', linewidth=4)
    plt.plot(predicted_output, '--', label='predicted output')
    plt.xlabel('n')
    plt.ylabel('y_n')
    plt.grid()
    plt.legend()
    plt.show()
