import numpy as np
from SystemDefinition import TankSystem
import DataPreprocessing.constants as c


class DataGenerator:
    def __init__(self, system: str, batch_size=1):
        if system == 'TankSystem':
            self.system = TankSystem(parameters=c.TANK_SYSTEM)
        self.batch_size = batch_size
        self.input_sequence = None
        self.output_sequence = None

    def generate_data(self, input_sequence):
        self.input_sequence = input_sequence
        self.output_sequence = self.system.response(input_sequence)

    def get_raw_data(self):
        return self.input_sequence, self.output_sequence

    def get_data_for_neural_network(self):
        number_of_samples = np.shape(self.input_sequence)[0]
        number_of_inputs = np.shape(self.input_sequence)[1]
        input_sequence = np.reshape(self.input_sequence, newshape=(number_of_samples, 1, number_of_inputs))
        return input_sequence, self.output_sequence
