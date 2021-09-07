from neural_network import NewGRU, NewGRU2
from data_generator import ToyProblem, SimpleSilverbox, ToyProblemV2, ToyProblemV3
from simulation import Simulation
from multiprocessing import Process
import os
MAX_TASK = os.cpu_count() // 2
SIMULATION_REPEAT = 2
SIMULATION_LIST = [dict(data_generator=ToyProblem, number_of_train_samples=1000,
                        dataset_name='toy_problem_v1', new_layer=NewGRU2,
                        structure_file_name='structure_1.json',
                        number_of_epochs=20, batch_size=100, shuffle=True) for _ in range(SIMULATION_REPEAT)] + \
                  [dict(data_generator=ToyProblemV2, number_of_train_samples=1000,
                        dataset_name='toy_problem_v2', new_layer=NewGRU2,
                        structure_file_name='structure_1.json',
                        number_of_epochs=20, batch_size=100, shuffle=True) for _ in range(SIMULATION_REPEAT)] + \
                  [dict(data_generator=ToyProblemV3, number_of_train_samples=1000,
                        dataset_name='toy_problem_v3', new_layer=NewGRU2,
                        structure_file_name='structure_1.json',
                        number_of_epochs=20, batch_size=100, shuffle=True) for _ in range(SIMULATION_REPEAT)] + \
                  [dict(data_generator=ToyProblem, number_of_train_samples=1000,
                        dataset_name='toy_problem_v1', new_layer=NewGRU,
                        structure_file_name='structure_1.json',
                        number_of_epochs=20, batch_size=100, shuffle=True) for _ in range(SIMULATION_REPEAT)] + \
                  [dict(data_generator=ToyProblemV2, number_of_train_samples=1000,
                        dataset_name='toy_problem_v2', new_layer=NewGRU,
                        structure_file_name='structure_1.json',
                        number_of_epochs=20, batch_size=100, shuffle=True) for _ in range(SIMULATION_REPEAT)] + \
                  [dict(data_generator=ToyProblemV3, number_of_train_samples=1000,
                        dataset_name='toy_problem_v3', new_layer=NewGRU,
                        structure_file_name='structure_1.json',
                        number_of_epochs=20, batch_size=100, shuffle=True) for _ in range(SIMULATION_REPEAT)]


# TODO: Implement metric from article: Learning nonlinear state–space models using autoencoders
def sim(simulation_number, kwargs):
    if kwargs['data_generator'] == ToyProblem:
        data_generator = ToyProblem(number_of_train_samples=kwargs['number_of_train_samples'],
                                    number_of_test_samples=100, dataset_name=kwargs['dataset_name'], random_seed=False)
    elif kwargs['data_generator'] == ToyProblemV2:
        data_generator = ToyProblemV2(number_of_train_samples=kwargs['number_of_train_samples'],
                                      number_of_test_samples=100, dataset_name=kwargs['dataset_name'],
                                      random_seed=False)
    elif kwargs['data_generator'] == ToyProblemV3:
        data_generator = ToyProblemV3(number_of_train_samples=kwargs['number_of_train_samples'],
                                      number_of_test_samples=100, dataset_name=kwargs['dataset_name'], random_seed=False)
    elif kwargs['data_generator'] == SimpleSilverbox:
        data_generator = SimpleSilverbox(number_of_train_samples=kwargs['number_of_train_samples'],
                                         number_of_test_samples=100, dataset_name=kwargs['dataset_name'],
                                         train_phase=simulation_number % 10, test_phase=(simulation_number + 1) % 10)

    simulation = Simulation(data_generator=data_generator, new_layer=kwargs['new_layer'],
                            structure_file_name=kwargs['structure_file_name'],
                            simulation_number=simulation_number)
    simulation.run(number_of_epochs=kwargs['number_of_epochs'], batch_size=kwargs['batch_size'],
                   shuffle=kwargs['shuffle'])


if __name__ == '__main__':
    processes = []
    for index, sim_parameters in enumerate(SIMULATION_LIST):
        while len(processes) == MAX_TASK:
            for process in processes.copy():
                process.join(timeout=0)
                if not process.is_alive():
                    processes.remove(process)
        p = Process(target=sim, args=(index + 34, sim_parameters))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
