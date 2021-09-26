import numpy as np
import os
import json
import matplotlib.pyplot as plt
from layers import NewGRU, NewGRULinear
from models import NeuralNetworkNewGRU
from data_generator import ToyProblemOriginal, ToyProblemComplexRoots, ToyProblemRealRoots


STRUCTURE_PATH = os.path.abspath('../simulation/neural_network_structure/')


def true_system_weight():
    generator = ToyProblemRealRoots()
    structure_file_path = os.path.join(STRUCTURE_PATH, 'structure_2.json')
    if os.path.exists(structure_file_path) is False:
        raise FileNotFoundError('File not exist.')
    with open(structure_file_path, 'r') as file:
        structure = json.load(file)
    number_of_inputs = 1
    number_of_outputs = 1
    neural_network = NeuralNetworkNewGRU(structure=structure, layer=NewGRULinear,
                                         number_of_inputs=number_of_inputs, number_of_outputs=number_of_outputs)
    neural_network.reset_state()
    weights = neural_network.get_weights()
    weights[0] = np.concatenate([generator.system.input_matrix, generator.system.state_matrix], axis=-1).transpose()
    weights[1] = np.zeros((2,))
    weights[2] = np.reshape(generator.system.output_matrix, newshape=(2, 1))
    weights[3] = np.zeros(1, )
    neural_network.set_weights(weights)
    impulse_sequence = np.zeros((100, 1, number_of_inputs))
    impulse_sequence[0, :, :] = 1
    outputs = neural_network.predict_sequence(inputs=impulse_sequence)
    plt.plot(outputs)
    plt.show()


def trained_weights():
    generator = ToyProblemRealRoots()
    structure_file_path = os.path.join(STRUCTURE_PATH, 'structure_2.json')
    if os.path.exists(structure_file_path) is False:
        raise FileNotFoundError('File not exist.')
    with open(structure_file_path, 'r') as file:
        structure = json.load(file)
    number_of_inputs = 1
    number_of_outputs = 1
    neural_network = NeuralNetworkNewGRU(structure=structure, layer=NewGRULinear,
                                         number_of_inputs=number_of_inputs, number_of_outputs=number_of_outputs)
    neural_network.reset_state()
    neural_network.load_weights(file_path='test.h5')
    impulse_sequence = np.zeros((100, 1, number_of_inputs))
    impulse_sequence[0, :, :] = 1
    outputs = neural_network.predict_sequence(inputs=impulse_sequence)
    plt.plot(outputs)
    plt.show()


if __name__ == '__main__':
    true_system_weight()
    trained_weights()
