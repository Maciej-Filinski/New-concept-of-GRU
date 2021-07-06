from NeuralNetwork import NewGRU
import tensorflow as tf
from tensorflow import keras
import constants as c


def build_model(ss_vector_length, number_of_inputs, batch_size):
    cell = NewGRU(state_length=c.NEURAL_NETWORK_STATE_LENGTH,
                  number_of_outputs=ss_vector_length,
                  candidate_dnn_structure=c.CANDIDATE_DNN_STRUCTURE,
                  forget_dnn_structure=c.FORGET_DNN_STRUCTURE,
                  output_dnn_structure=c.OUTPUT_DNN_STRUCTURE,
                  forget_dnn_enable=False,
                  output_dnn_enable=False)
    rnn = keras.layers.RNN(cell, stateful=True)
    input_layer = keras.Input(batch_shape=(batch_size, None, number_of_inputs), ragged=True)
    output = rnn(input_layer)
    model = keras.models.Model(input_layer, output)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.metrics.RootMeanSquaredError())
    return model
