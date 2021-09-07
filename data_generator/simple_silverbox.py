from .data_generator import DataGenerator
from system_definition import SimpleSilverboxSystem
import numpy as np


class SimpleSilverbox(DataGenerator):
    number_of_inputs = 1
    state_length = 2
    number_of_outputs = 1
    scale = 1

    def __init__(self,
                 number_of_train_samples: int,
                 number_of_test_samples: int,
                 dataset_name: str,
                 train_phase: int = 1,
                 test_phase: int = 2):
        self.system = SimpleSilverboxSystem()
        self.train_phase = train_phase
        self.test_phase = test_phase
        super().__init__(dataset_name, number_of_train_samples, number_of_test_samples, create=True)

    def _create_data(self):
        print('Loading...')
        data = np.load(self.data_path_load)
        self.data = {'train': {'inputs': data['inputs'][:self.number_of_train_samples, self.train_phase],
                               'outputs': np.reshape(data['outputs'][:self.number_of_train_samples, self.train_phase],
                                                     newshape=(self.number_of_train_samples, self.number_of_outputs))},
                     'test': {'inputs': data['inputs'][:self.number_of_test_samples, self.test_phase],
                              'outputs': np.reshape(data['outputs'][:self.number_of_test_samples, self.test_phase],
                                                    newshape=(self.number_of_test_samples, self.number_of_outputs))}}
