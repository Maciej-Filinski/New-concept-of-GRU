import os
import json
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = os.path.abspath('../NewConceptOfGRU/simulation/result')
MODEL_DIR = os.path.abspath('../NewConceptOfGRU/simulation/trained_model')


class Result:
    def __init__(self, problem_name: str, structure: dict, number_of_train_sample: int,
                 simulation_number: int):
        self.problem_name = problem_name
        self.structure = structure
        self.number_of_train_sample = number_of_train_sample
        self.batch_size = None
        self.simulation_number = simulation_number
        self.result_path = os.path.join(RESULT_DIR, self.problem_name + '_sim_' + str(simulation_number) + '.json')
        self.plot_path = os.path.join(RESULT_DIR, self.problem_name + '_sim_' + str(simulation_number) + '.npz')
        self.model_path = os.path.join(MODEL_DIR, self.problem_name + '_sim_' + str(simulation_number) + '.hdf5')
        self.data = None

    def save(self):
        information = {'problem name': self.problem_name,
                       'structure': self.structure,
                       'batch size': self.batch_size,
                       'number of train sample': self.number_of_train_sample,
                       'trained model path': self.model_path}
        with open(self.result_path, 'w') as file:
            json.dump(information, file, indent=4)
        np.savez(self.plot_path, **self.data)

    def load(self, path):
        self.result_path = path + '.json'
        self.plot_path = path + '.npz'
        self.model_path = path + '.hdf5'
        with open(self.result_path) as file:
            data = json.load(file)
        self.problem_name = data['problem name']
        self.structure = data['structure']
        self.batch_size = data['batch size']
        self.number_of_train_sample = data['number of train sample']
        self.model_path = data['trained model path']
        data = np.load(self.plot_path)
        self.data = {'loss_function': data['loss_function'],
                     'execution_time': data['execution_time'],
                     'train_inputs': data['train_inputs'],
                     'train_outputs': data['train_outputs'],
                     'train_predicted_outputs': data['train_predicted_outputs'],
                     'train_network_state': data['train_network_state'],
                     'train_forget': data['train_forget'],
                     'train_candidate': data['train_candidate'],
                     'test_inputs': data['test_inputs'],
                     'test_outputs': data['test_outputs'],
                     'test_predicted_outputs': data['test_predicted_outputs'],
                     'test_network_state': data['test_network_state'],
                     'test_forget': data['test_forget'],
                     'test_candidate': data['test_candidate']}

    def plot(self):
        if self.data is not None:
            # fig_1, axs_1 = plt.subplots(1, 2)
            # x = np.arange(self.data['loss_function'].shape[1]) + 1
            # for i in range(self.data['loss_function'].shape[0]):
            #     axs_1[0].plot(x, self.data['loss_function'][i, :], label='epoch '+str(i+1))
            # axs_1[0].set_ylabel('loss')
            # axs_1[0].set_xlabel('step')
            # axs_1[0].legend()
            # axs_1[0].grid()
            # for i in range(self.data['execution_time'].shape[0]):
            #     axs_1[1].plot(x, self.data['execution_time'][i, :], label='epoch '+str(i+1))
            # axs_1[1].set_ylabel('time [s]')
            # axs_1[1].set_xlabel('step')
            # axs_1[1].legend()
            # axs_1[1].grid()

            model_state_length = self.data['train_network_state'].shape[0]
            fig, axs = plt.subplots(4, 2)
            axs[0, 0].plot(self.data['train_outputs'], 'g', label='true output', linewidth=2)
            axs[0, 0].plot(self.data['train_predicted_outputs'], '--r', label='predicted output', linewidth=2)
            axs[0, 0].set_title('TRAIN')
            # axs[0, 0].legend()
            axs[0, 0].set_ylim([-1, 1])
            axs[0, 0].grid()
            axs[0, 0].set_ylabel(r'$\hat{y}_n$')

            axs[0, 1].plot(self.data['test_outputs'], 'g', label='true output', linewidth=2)
            axs[0, 1].plot(self.data['test_predicted_outputs'], '--r', label='predicted output', linewidth=2)
            axs[0, 1].set_title('TEST')
            axs[0, 1].set_ylim([-1, 1])
            axs[0, 1].grid()
            # axs[0, 1].legend()

            axs[1, 0].plot(self.data['train_network_state'], '--', label='neural network state', linewidth=2)
            axs[1, 0].legend()
            axs[1, 0].set_ylabel(r'$h_n$')
            axs[2, 0].plot(self.data['train_candidate'], linewidth=2)
            axs[2, 0].legend(['candidate ' + str(i + 1) for i in range(model_state_length)])
            axs[2, 0].set_ylabel(r'$\hat{h}_n$')
            axs[3, 0].plot(self.data['train_forget'], linewidth=2)
            axs[3, 0].legend(['forget ' + str(i + 1) for i in range(model_state_length)])
            axs[3, 0].set_ylabel(r'$f_n$')
            axs[3, 0].set_ylim([0, 1])

            axs[1, 1].plot(self.data['test_network_state'], '--', label='neural network state', linewidth=2)
            axs[1, 1].legend()
            axs[2, 1].plot(self.data['test_candidate'], linewidth=2)
            axs[2, 1].legend(['candidate ' + str(i + 1) for i in range(model_state_length)])
            axs[3, 1].plot(self.data['test_forget'], linewidth=2)
            axs[3, 1].legend(['forget ' + str(i + 1) for i in range(model_state_length)])
            axs[3, 0].set_xlabel(r'$n$')
            axs[3, 1].set_xlabel(r'$n$')
            axs[3, 1].set_ylim([0, 1])

            plt.show()
        else:
            print('Result no available.')
