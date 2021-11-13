from .data_generator import DataGenerator
from system_definition import ToyProblemSystemOriginal, ToyProblemSystemComplexRoots, ToyProblemSystemRealRoots
import numpy as np


class ToyProblemOriginal(DataGenerator):
    number_of_inputs = 1
    number_of_outputs = 1
    scale = 1

    def __init__(self,
                 number_of_train_samples: int = 0,
                 number_of_test_samples: int = 0,
                 dataset_name: str = '',
                 random_seed: bool = False):
        self.system = ToyProblemSystemOriginal()
        self.random_seed = random_seed
        if random_seed is True:
            super().__init__('toy_problem_random_seed', number_of_train_samples, number_of_test_samples, create=True)
        else:
            super().__init__(dataset_name, number_of_train_samples, number_of_test_samples, create=False)

    def _create_data(self):
        print('Create data...')
        if self.random_seed is False:
            np.random.seed(1)
        inputs = np.random.uniform(-1, 1, size=(self.number_of_train_samples, 1))
        self.data['train'] = self.system.response(inputs=inputs)
        inputs = np.reshape(0.5 * np.sin(2 * np.pi * np.arange(self.number_of_test_samples) / 50),
                            newshape=(self.number_of_test_samples, 1))
        self.data['test'] = self.system.response(inputs=inputs)
        self._save_data()


class ToyProblemComplexRoots(ToyProblemOriginal):
    def __init__(self,
                 number_of_train_samples: int = 0,
                 number_of_test_samples: int = 0,
                 dataset_name: str = '',
                 random_seed: bool = False):
        super().__init__(number_of_train_samples, number_of_test_samples, dataset_name,  random_seed=random_seed)
        self.system = ToyProblemSystemComplexRoots()


class ToyProblemRealRoots(ToyProblemOriginal):
    def __init__(self,
                 number_of_train_samples: int = 0,
                 number_of_test_samples: int = 0,
                 dataset_name: str = '',
                 random_seed: bool = False):
        super().__init__(number_of_train_samples, number_of_test_samples, dataset_name,  random_seed=random_seed)
        self.system = ToyProblemSystemRealRoots()

