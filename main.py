from layers import NewGRU, NewGRULinear
from data_generator import ToyProblemOriginal, SimpleSilverbox, ToyProblemComplexRoots, ToyProblemRealRoots
from simulation import SimulationNewGRU
from multiprocessing import Process
import os


SIMULATION_REPEAT = 1
SIMULATION_LIST = [dict(data_generator=ToyProblemOriginal, number_of_train_samples=100,
                        dataset_name='toy_problem', new_layer=NewGRULinear,
                        structure_file_name='structure_2.json',
                        number_of_epochs=500, batch_size=100, shuffle=True) for _ in range(SIMULATION_REPEAT)]


# TODO: Implement metric from article: Learning nonlinear state–space models using autoencoders
def sim(simulation_number, kwargs):
    if kwargs['data_generator'] == ToyProblemOriginal:
        data_generator = ToyProblemOriginal(number_of_train_samples=kwargs['number_of_train_samples'],
                                            number_of_test_samples=100, dataset_name=kwargs['dataset_name'], random_seed=False)
    elif kwargs['data_generator'] == ToyProblemComplexRoots:
        data_generator = ToyProblemComplexRoots(number_of_train_samples=kwargs['number_of_train_samples'],
                                                number_of_test_samples=100, dataset_name=kwargs['dataset_name'],
                                                random_seed=False)
    elif kwargs['data_generator'] == ToyProblemRealRoots:
        data_generator = ToyProblemRealRoots(number_of_train_samples=kwargs['number_of_train_samples'],
                                             number_of_test_samples=100, dataset_name=kwargs['dataset_name'], random_seed=False)
    elif kwargs['data_generator'] == SimpleSilverbox:
        data_generator = SimpleSilverbox(number_of_train_samples=kwargs['number_of_train_samples'],
                                         number_of_test_samples=100, dataset_name=kwargs['dataset_name'],
                                         train_phase=simulation_number % 10, test_phase=(simulation_number + 1) % 10)

    simulation = SimulationNewGRU(data_generator=data_generator, new_layer=kwargs['new_layer'],
                                  structure_file_name=kwargs['structure_file_name'],
                                  simulation_number=simulation_number)
    simulation.run(number_of_epochs=kwargs['number_of_epochs'], batch_size=kwargs['batch_size'],
                   shuffle=kwargs['shuffle'])


if __name__ == '__main__':
    processes = []
    max_task = min(os.cpu_count(), len(SIMULATION_LIST))
    for index, sim_parameters in enumerate(SIMULATION_LIST):
        while len(processes) == max_task:
            for process in processes.copy():
                process.join(timeout=0)
                if not process.is_alive():
                    processes.remove(process)
        p = Process(target=sim, args=(index, sim_parameters))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
    