import tensorflow as tf
from tensorflow import keras
import numpy as np


class NeuralNetworkNewGRU:
    def __init__(self, structure, layer, number_of_inputs, number_of_outputs):
        self.layer = layer
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.model = self._build_model(structure, layer, number_of_inputs, number_of_outputs)

    def predict_one_step(self, inputs):
        return self.model(inputs)

    def predict_sequence(self, inputs):
        sequence_length = len(inputs)
        outputs = np.zeros((sequence_length, self.number_of_outputs))
        for i in range(len(inputs)):
            outputs[i, :] = self.model(inputs[i, :, :])
        return outputs

    def reset_state(self):
        self.model.reset_states()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def load_weights(self, file_path: str):
        self.model.load_weights(file_path)

    def save_weights(self, file_path: str):
        self.model.save_weights(file_path)

    @staticmethod
    def _build_model(structure, layer, number_of_inputs, number_of_outputs):
        cell = layer(state_length=structure['state_length'],
                     number_of_outputs=number_of_outputs,
                     candidate_dnn_structure=structure['candidate_dnn'],
                     forget_dnn_structure=structure['forget_dnn'],
                     output_dnn_structure=structure['output_dnn'],
                     forget_dnn_enable=structure['forget_dnn_enable'],
                     output_dnn_enable=structure['output_dnn_enable'],
                     return_full_output=structure['return_full_output'])
        rnn = keras.layers.RNN(cell, stateful=True)
        input_layer = keras.Input(batch_shape=(1, None, number_of_inputs), ragged=True)
        output = rnn(input_layer)
        model = keras.models.Model(input_layer, output)
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
        model.summary()
        return model
