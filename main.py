from neural_network import NewGRU
from data_generator import ToyProblem
from simulation import Simulation

# TODO: Implement metric from article: Learning nonlinear state–space models using autoencoders

if __name__ == '__main__':
    toy_problem = ToyProblem(number_of_train_samples=100, number_of_test_samples=100, random_seed=False)
    simulation = Simulation(data_generator=toy_problem, new_layer=NewGRU, structure_file_name='structure_1.json')
    simulation.run(number_of_epochs=1, batch_size=100, shuffle=True)
    simulation.result.plot()
