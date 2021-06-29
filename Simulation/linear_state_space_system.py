from NeuralNetwork import NewGRU
import constants as c
import tensorflow as tf
from tensorflow import keras


def create_neural_network(number_of_inputs: int, number_of_outputs: int):
    cell = NewGRU(state_length=c.NEURAL_NETWORK_STATE_LENGTH,
                  number_of_outputs=number_of_outputs,
                  input_dnn_structure=c.INPUT_DNN_STRUCTURE,
                  forget_dnn_structure=c.FORGET_DNN_STRUCTURE,
                  output_dnn_structure=c.OUTPUT_DNN_STRUCTURE,
                  forget_dnn_enable=True,
                  output_dnn_enable=True)
    rnn = keras.layers.RNN(cell, stateful=True)
    input_layer = keras.Input(batch_shape=(1, None, number_of_inputs))
    output = rnn(input_layer)
    model = keras.models.Model(input_layer, output)
    model.compile(optimizer='adam', loss='mse', metrics=tf.keras.metrics.RootMeanSquaredError())
    model.summary()
    return model


if __name__ == '__main__':
    neural_network = create_neural_network(number_of_inputs=1, number_of_outputs=1)
