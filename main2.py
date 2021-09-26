import os
import json
from layers import NewGRU, NewGRULinear
from simulation import Simulation
from data_generator import ToyProblemOriginal, ToyProblemRealRoots, ToyProblemComplexRoots
from models import NeuralNetworkNewGRU


def build_simulation_list_from(files_list):
    simulation_list = []
    structure_path = os.path.abspath('./simulation/neural_network_structure')
    for file in files_list:
        with open(file) as f:
            sim = json.load(f)
        data_generator = None
        neural_network = None
        if sim['data_generator_name'] == 'toy_problem_original':
            data_generator = ToyProblemOriginal(sim['data_generator']['number_of_train_samples'],
                                                sim['data_generator']['number_of_test_samples'],
                                                sim['data_generator']['data_file_name'])
        if sim['data_generator_name'] == 'toy_problem_real_roots':
            data_generator = ToyProblemRealRoots(sim['data_generator']['number_of_train_samples'],
                                                 sim['data_generator']['number_of_test_samples'],
                                                 sim['data_generator']['data_file_name'])
        if sim['data_generator_name'] == 'toy_problem_complex_roots':
            data_generator = ToyProblemComplexRoots(sim['data_generator']['number_of_train_samples'],
                                                    sim['data_generator']['number_of_test_samples'],
                                                    sim['data_generator']['data_file_name'])
        with open(os.path.join(structure_path, sim['structure_file'])) as file:
            structure = json.load(file)
        if sim['layer_type'] == 'newGRU':
            neural_network = NeuralNetworkNewGRU(structure, NewGRU, number_of_inputs=1, number_of_outputs=1)
        if sim['layer_type'] == 'Linear_newGRU':
            neural_network = NeuralNetworkNewGRU(structure, NewGRULinear, number_of_inputs=1, number_of_outputs=1)
        simulation = Simulation(data_generator, neural_network, simulation_number=1)
        simulation.set_param_for_train(sim['number_of_epochs'], sim['batch_size'][0], shuffle=sim['shuffle'])
        for i in range(sim['number_of_repetitions']):
            simulation_list.append(simulation)
    return simulation_list


if __name__ == '__main__':
    simulation_path = os.path.abspath('./simulation/simulation_list')
    files = os.listdir(simulation_path)
    files = [os.path.join(simulation_path, file) for file in files if file.endswith('.json') and file.startswith('simulation_')]
    simulation = build_simulation_list_from(files_list=files)
    print(simulation)
    simulation[0].run()
