from NeuralNetwork import NewGRU
import constants as c
import tensorflow as tf
from tensorflow import keras
from DataPreprocessing import DataGenerator
import numpy as np
import matplotlib.pyplot as plt


def create_neural_network(number_of_inputs: int, number_of_outputs: int):
    cell = NewGRU(state_length=c.NEURAL_NETWORK_STATE_LENGTH,
                  number_of_outputs=number_of_outputs,
                  input_dnn_structure=c.INPUT_DNN_STRUCTURE,
                  forget_dnn_structure=c.FORGET_DNN_STRUCTURE,
                  output_dnn_structure=c.OUTPUT_DNN_STRUCTURE,
                  forget_dnn_enable=False,
                  output_dnn_enable=True)
    rnn = keras.layers.RNN(cell, stateful=True)
    input_layer = keras.Input(batch_shape=(1, None, number_of_inputs))
    output = rnn(input_layer)
    model = keras.models.Model(input_layer, output)
    model.compile(optimizer='adam', loss='mse', metrics=tf.keras.metrics.RootMeanSquaredError())
    model.summary()
    return model


if __name__ == '__main__':
    number_of_samples = 2500
    train_data_length = 2000

    neural_network = create_neural_network(number_of_inputs=1, number_of_outputs=1)
    data_generator = DataGenerator(system='TankSystem')
    input_sequence = np.random.uniform(1, 2, (number_of_samples, 1))
    data_generator.generate_data(input_sequence=input_sequence)
    all_input_data, all_output_data = data_generator.get_data_for_neural_network()
    train_input_sequence = all_input_data[:train_data_length, :, :]
    validate_input_sequence = all_input_data[train_data_length:, :, :]
    train_output_sequence = all_output_data[:train_data_length, :]
    validate_output_sequence = all_output_data[train_data_length:, :]

    print(train_output_sequence)
    print(train_input_sequence)

    neural_network.fit(train_input_sequence, train_output_sequence,
                       epochs=50,
                       batch_size=1,
                       validation_data=(validate_input_sequence, validate_output_sequence))

    test_data_length = 200
    input_sequence = 0.5 * np.sin(2 * np.pi * np.array(range(test_data_length)) / 10) + 1.5
    input_sequence = np.reshape(input_sequence, newshape=(test_data_length, 1))
    data_generator.generate_data(input_sequence=input_sequence)
    test_input_sequence, test_output_sequence = data_generator.get_data_for_neural_network()
    neural_network.reset_states()
    predicted_output = np.zeros(shape=(test_data_length, 1))
    for i in range(test_data_length):
        predicted_output[i, :] = neural_network.predict(test_input_sequence[i, :, :])
    print(predicted_output)
    plt.plot(test_output_sequence)
    plt.plot(predicted_output)
    plt.show()