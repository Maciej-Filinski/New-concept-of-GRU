import os
import json


if __name__ == '__main__':
    path = os.path.abspath('./')
    files = os.listdir(path)
    files = [file for file in files if file.endswith('.json')]
    for file in files:
        with open(file) as f:
            simulation = json.load(f)
        print(f'File: {file}')
        print(f'    Simulation description: {simulation["description"]}')
        print(f'    Layer type: {simulation["layer_type"]}')
        print(f'    Data generator:')
        print(f'        Data file name: {simulation["data_generator"]["data_file_name"]}')
        print(f'        Number of train samples: {simulation["data_generator"]["number_of_train_samples"]}')
        print(f'        Number of test samples: {simulation["data_generator"]["number_of_test_samples"]}')
        print(f'        Random seed: {simulation["data_generator"]["random_seed"]}')
        print(f'    Number of repetitions: {simulation["number_of_repetitions"]}')
        print(f'    Number of epochs: {simulation["number_of_epochs"]}')
        print(f'    Batch size: {simulation["batch_size"]}')

