import numpy as np


class TankSystem:
    def __init__(self, parameters):
        self.k_1 = parameters['k_1']
        self.k_2 = parameters['k_2']
        self.k_3 = parameters['k_3']
        self.k_4 = parameters['k_4']
        self.control_function = parameters['control_function']
        self.number_of_inputs = 1
        self.number_of_outputs = 1

    def response(self, input_sequence, initial_state=np.array([0.0, 0.0])):
        number_of_samples = np.shape(input_sequence)[0]
        state = initial_state
        output_sequence = np.zeros(shape=np.shape(input_sequence))
        for i in range(number_of_samples):
            state[0] = state[0] - self.k_1 * np.sqrt(state[0]) + self.k_2 * self.control_function(input_sequence[i])
            state[1] = state[1] + self.k_3 * np.sqrt(state[0]) - self.k_4 * np.sqrt(state[1])
            output_sequence[i] = state[1]
        return output_sequence
