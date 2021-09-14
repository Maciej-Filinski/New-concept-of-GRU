from .data_generator import DataGenerator
from system_definition import ToyProblemSystem, ToyProblemSystemV2, ToyProblemSystemV3
import numpy as np


class ToyProblem(DataGenerator):
    number_of_inputs = 1
    number_of_outputs = 1
    scale = 1

    def __init__(self,
                 number_of_train_samples: int,
                 number_of_test_samples: int,
                 dataset_name: str,
                 random_seed=False):
        self.system = ToyProblemSystem()
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


class ToyProblemV2(ToyProblem):
    def __init__(self,
                 number_of_train_samples: int,
                 number_of_test_samples: int,
                 dataset_name: str,
                 random_seed=False):
        super().__init__(number_of_train_samples, number_of_test_samples, dataset_name,  random_seed=random_seed)
        self.system = ToyProblemSystemV2()


class ToyProblemV3(ToyProblem):
    def __init__(self,
                 number_of_train_samples: int,
                 number_of_test_samples: int,
                 dataset_name: str,
                 random_seed=False):
        super().__init__(number_of_train_samples, number_of_test_samples, dataset_name,  random_seed=random_seed)
        self.system = ToyProblemSystemV3()

