import json
import os


NEURAL_NETWORK_STATE_LENGTH = 5
FILE_NAME = 'simulation_1.json'

SIMULATION = {'description': 'Learning neural network with layer basic new GRU using data from basic toy problem',
              'simulation_type': 'newGRU',
              'layer_type': 'newGRU',
              'structure_file': 'structure_1.json',
              'data_generator_name': 'toy_problem_original',
              'data_generator': {'data_file_name': 'toy_problem_v1',
                                 'number_of_train_samples': 1000,
                                 'number_of_test_samples': 300
                                 },
              'number_of_epochs': 100,
              'batch_size': [20],
              'shuffle': True,
              'number_of_repetitions': 100,
              }


if __name__ == '__main__':
    file_path = os.path.join(os.path.abspath('../simulation_list'), FILE_NAME)
    if os.path.exists(file_path) is True:
        decision = input('Overwrite the file? Yes/no: ')
        if decision == '':
            decision = 'yes'
    else:
        decision = 'yes'
    if decision.lower() == 'yes':
        with open(file_path, 'w') as file:
            json.dump(SIMULATION, file, indent=4)
        print('The data has been saved.')
    else:
        print('The data has not been saved.')