import tensorflow as tf
from tensorflow import keras
import numpy as np


class AbstractModel:
    def __init__(self, structure, layer, number_of_inputs, number_of_outputs):
        self.layer = layer
        self.structure = structure
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.model = self.build_model(structure, layer, number_of_inputs, number_of_outputs)

    def predict_one_step(self, inputs):
        return self.model(inputs)

    def predict_sequence(self, inputs):
        sequence_length = len(inputs)
        outputs = np.zeros((sequence_length, self.number_of_outputs))
        for i in range(len(inputs)):
            outputs[i, :] = self.model(inputs[i, :, :])
        return outputs

    def reset_states(self):
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
    def build_model(structure, layer, number_of_inputs, number_of_outputs):
        raise NotImplementedError('Method build_model not implemented')
