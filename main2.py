import os
import json
import time
from multiprocessing import Process
from layers import NewGRU, NewGRULinear
from simulation import Simulation
from data_generator import ToyProblemOriginal, ToyProblemRealRoots, ToyProblemComplexRoots
from models.model_newGRU import build_new_gru_for_train, build_new_gru_for_test


def build_simulation_list_from(files_list):
    simulation_list = []
    for simulation_file in files_list:
        with open(simulation_file) as file:
            sim = json.load(file)
        for i in range(sim['number_of_repetitions']):
            simulation_list.append(sim)

    return simulation_list


def run(simulation_params, simulation_number):
    structure_path = os.path.abspath('./simulation/neural_network_structure')
    if simulation_params['data_generator_name'] == 'toy_problem_original':
        data_generator = ToyProblemOriginal(simulation_params['data_generator']['number_of_train_samples'],
                                            simulation_params['data_generator']['number_of_test_samples'],
                                            simulation_params['data_generator']['data_file_name'])
    elif simulation_params['data_generator_name'] == 'toy_problem_real_roots':
        data_generator = ToyProblemRealRoots(simulation_params['data_generator']['number_of_train_samples'],
                                             simulation_params['data_generator']['number_of_test_samples'],
                                             simulation_params['data_generator']['data_file_name'])
    elif simulation_params['data_generator_name'] == 'toy_problem_complex_roots':
        data_generator = ToyProblemComplexRoots(simulation_params['data_generator']['number_of_train_samples'],
                                                simulation_params['data_generator']['number_of_test_samples'],
                                                simulation_params['data_generator']['data_file_name'])
    else:
        data_generator = None
    with open(os.path.join(structure_path, simulation_params['structure_file'])) as file:
        structure = json.load(file)
    if simulation_params['layer_type'] == 'newGRU':
        model_for_train = build_new_gru_for_train(structure=structure,
                                                  layer=NewGRU,
                                                  number_of_inputs=data_generator.number_of_inputs,
                                                  number_of_outputs=data_generator.number_of_outputs)
        model_for_test = build_new_gru_for_test(structure=structure,
                                                layer=NewGRU,
                                                number_of_inputs=data_generator.number_of_inputs,
                                                number_of_outputs=data_generator.number_of_outputs)
    elif simulation_params['layer_type'] == 'Linear_newGRU':
        model_for_train = build_new_gru_for_train(structure=structure,
                                                  layer=NewGRULinear,
                                                  number_of_inputs=data_generator.number_of_inputs,
                                                  number_of_outputs=data_generator.number_of_outputs)
        model_for_test = build_new_gru_for_test(structure=structure,
                                                layer=NewGRULinear,
                                                number_of_inputs=data_generator.number_of_inputs,
                                                number_of_outputs=data_generator.number_of_outputs)
    else:
        model_for_train = None
        model_for_test = None
    simulation = Simulation(data_generator=data_generator,
                            model_for_train=model_for_train,
                            model_for_test=model_for_test,
                            structure=structure,
                            simulation_number=simulation_number)
    simulation.set_param_for_train(simulation_params['number_of_epochs'],
                                   simulation_params['batch_size'],
                                   shuffle=simulation_params['shuffle'])
    simulation.run()


if __name__ == '__main__':
    simulation_path = os.path.abspath('./simulation/simulation_list')
    files = os.listdir(simulation_path)
    files = [os.path.join(simulation_path, file) for file in files
             if file.endswith('.json') and file.startswith('simulation_')]
    simulations = build_simulation_list_from(files_list=files)
    print(f'Simulation to run = {len(simulations)}')

    processes = []
    max_task = min(os.cpu_count(), len(simulations))
    for index, sim in enumerate(simulations):
        while len(processes) == max_task:
            for process in processes.copy():
                process.join(timeout=0)
                if not process.is_alive():
                    processes.remove(process)
        p = Process(target=run, args=(sim, index))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
