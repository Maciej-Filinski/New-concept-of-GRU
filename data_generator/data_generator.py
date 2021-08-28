import os
import numpy as np


class DataGenerator:
    number_of_inputs = None
    state_length = None
    number_of_outputs = None
    scale = 1

    def __init__(self,
                 data_file_name: str,
                 number_of_train_samples: int,
                 number_of_test_samples: int,
                 create: bool = False):
        self.__name__ = data_file_name
        self.data_path_load = os.path.join(os.path.abspath('../NewConceptOfGRU/simulation/datasets'),
                                           data_file_name + '.npz')
        self.data_path_save = os.path.join(os.path.abspath('../NewConceptOfGRU/simulation/datasets'), data_file_name)
        self.number_of_test_samples = number_of_test_samples
        self.number_of_train_samples = number_of_train_samples
        self.create = create
        self.data = {}

    def load_data(self):
        if self.create is True:
            self._create_data()
            return self.data
        if os.path.exists(self.data_path_load) is True:
            print('File exist.')
            if os.path.getsize(self.data_path_load) != 0:
                print('Loading...')
                self._load_from_file()
                if self.data['train']['inputs'].shape[0] < self.number_of_train_samples:
                    print('Not enough sample in file.')
                    self._create_data()
                if self.data['test']['inputs'].shape[0] < self.number_of_test_samples:
                    self._create_data()
            else:
                print('File is empty.')
                self._create_data()
        else:
            print('File not exist.')
            self._create_data()

        print('Data prepared.')
        self.data['train']['inputs'] = np.reshape(self.data['train']['inputs'],
                                                  newshape=(self.number_of_train_samples, 1, self.number_of_inputs))
        self.data['test']['inputs'] = np.reshape(self.data['test']['inputs'],
                                                 newshape=(self.number_of_test_samples, 1, self.number_of_inputs))
        self.data['train']['outputs'] /= self.scale
        self.data['test']['outputs'] /= self.scale
        return self.data

    def _load_from_file(self):
        self.data = {}
        data = np.load(self.data_path_load)
        self.data['test'] = {key.replace('test', ''): data[key][: self.number_of_test_samples, :]
                             for key in data.keys() if key.startswith('test')}
        self.data['train'] = {key.replace('train', ''): data[key][: self.number_of_train_samples, :]
                              for key in data.keys() if key.startswith('train')}

    def _save_data(self):
        to_save = {}
        for key_1 in self.data.keys():
            for key_2 in self.data[key_1].keys():
                to_save[key_1 + key_2] = self.data[key_1][key_2]
        np.savez(self.data_path_save, **to_save)
        print('Data saved.')

    def _create_data(self):
        raise Exception('Not define create_data function in child class')
