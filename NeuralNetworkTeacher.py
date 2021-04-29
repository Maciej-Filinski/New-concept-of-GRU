from SystemDefinition import LinearSystem
import numpy as np


class TeacherLS:
    def __init__(self, system: LinearSystem, number_of_samples, initial_state, neural_network_input_length):
        self.system = system
        self.number_of_samples = number_of_samples
        self.initial_state = initial_state
        self.neural_network_input_length = neural_network_input_length
        self.input_sequence = None
        self.output_sequence = None
        self.neural_network_input_sequence = None
        self._generate_data()
        self._prepare_data_for_neural_network()
        pass

    def _generate_data(self):
        input_sequence = np.random.uniform(-1.73, 1.73, size=(self.number_of_samples, 1))
        output_sequence = self.system.linear_system_response(input_sequence=input_sequence,
                                                             initial_state=self.initial_state)
        self.input_sequence = input_sequence
        self.output_sequence = output_sequence[self.neural_network_input_length::, :]

    def _prepare_data_for_neural_network(self):
        number_of_inputs = np.shape(self.input_sequence)[1]
        neural_network_input_sequence = np.zeros(shape=(self.number_of_samples - self.neural_network_input_length,
                                                        self.neural_network_input_length,
                                                        number_of_inputs))
        for i in range(self.number_of_samples - self.neural_network_input_length):
            for j in range(self.neural_network_input_length):
                neural_network_input_sequence[i, :, :] = self.input_sequence[i: i + self.neural_network_input_length, :]
        self.neural_network_input_sequence = neural_network_input_sequence

    def get_data(self):
        self.output_sequence = 2 * self.output_sequence / (max(self.output_sequence) - min(self.output_sequence))
        return self.neural_network_input_sequence, self.output_sequence