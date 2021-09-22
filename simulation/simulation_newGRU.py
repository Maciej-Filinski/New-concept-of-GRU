from data_generator import DataGenerator
from simulation import Result
import os
import json
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import datetime


class SimulationNewGRU:
    def __init__(self, data_generator: DataGenerator, new_layer, structure_file_name: str, simulation_number: int = 1):
        self.data_generator = data_generator
        self.structure_file_name = os.path.join(os.path.abspath('../New-concept-of-GRU/simulation/neural_network_structure'),
                                                structure_file_name)
        print(self.structure_file_name)
        self.new_layer = new_layer
        if os.path.exists(self.structure_file_name) is False:
            raise FileNotFoundError('File not exist.')
        with open(self.structure_file_name, 'r') as file:
            self.structure = json.load(file)
        self.model = self._build_model(self.structure)
        self.result = Result(problem_name=self.data_generator.__name__,
                             structure=self.structure,
                             number_of_train_sample=self.data_generator.number_of_train_samples,
                             simulation_number=simulation_number)

    def run(self, number_of_epochs: int, batch_size: int, shuffle: bool = False):
        data = self.data_generator.load_data()
        indices_set = list(range(self.data_generator.number_of_train_samples))
        if shuffle is True:
            random.shuffle(indices_set)
        number_of_batch = self.data_generator.number_of_train_samples // batch_size
        execution_time = np.zeros(shape=(number_of_epochs, number_of_batch))
        loss_function = np.zeros(shape=(number_of_epochs, number_of_batch))
        for epoch in range(number_of_epochs):

            for step in range(number_of_batch):
                t_start = datetime.datetime.now()
                batch_shuffle_set = indices_set[step * batch_size: (step + 1) * batch_size]

                loss = self._run_one_step(x_train=data['train']['inputs'], y_train=data['train']['outputs'],
                                          batch_shuffle_set=batch_shuffle_set)

                t_end = datetime.datetime.now()
                execution_time[epoch, step] = (t_end - t_start).total_seconds()
                loss_function[epoch, step] = loss
                print(f'epoch:{epoch + 1:3}/{number_of_epochs:3}, step:{step + 1:3}/{number_of_batch:3},',
                      f'step execution time:{execution_time[epoch, step]:.2f}s,',
                      f'loss:{loss:.8f}')

        train_info = {'loss_function': loss_function,
                      'execution_time': execution_time}
        self.model.save_weights(self.result.model_path)
        weights = self.model.get_weights()
        self.structure['return_full_output'] = True
        self.model = self._build_model(self.structure)
        self.model.set_weights(weights)
        test_info = self._test(train_inputs=data['train']['inputs'][:self.data_generator.number_of_test_samples, :, :],
                               train_outputs=data['train']['outputs'][:self.data_generator.number_of_test_samples, :],
                               test_inputs=data['test']['inputs'],
                               test_outputs=data['test']['outputs'])
        self.result.data = {**train_info, **test_info}
        self.result.save()

    def _run_one_step(self, x_train: np.ndarray, y_train: np.ndarray, batch_shuffle_set: list):
        loss_function = tf.keras.losses.MeanSquaredError()

        max_value = np.max(batch_shuffle_set)
        self.model.reset_states()
        y_predicted = []
        with tf.GradientTape() as tape:
            if self.structure['return_full_output'] is True:
                for i in range(max_value + 1):
                    y, *rest = self.model(x_train[i, :, :])
                    y_predicted.append(y)
            else:
                for i in range(max_value + 1):
                    y = self.model(x_train[i, :, :])
                    y_predicted.append(y)
            y_predicted = tf.convert_to_tensor(y_predicted)
            y_1 = []
            y_2 = []
            for i in batch_shuffle_set:
                y_1.append(y_train[i, :])
                y_2.append(y_predicted[i, :, :])
            y_1 = tf.convert_to_tensor(y_1)
            y_2 = tf.convert_to_tensor(y_2)
            loss = loss_function(y_1, y_2)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def _build_model(self, structure):
        cell = self.new_layer(state_length=structure['state_length'],
                              number_of_outputs=self.data_generator.number_of_outputs,
                              candidate_dnn_structure=structure['candidate_dnn'],
                              forget_dnn_structure=structure['forget_dnn'],
                              output_dnn_structure=structure['output_dnn'],
                              forget_dnn_enable=structure['forget_dnn_enable'],
                              output_dnn_enable=structure['output_dnn_enable'],
                              return_full_output=structure['return_full_output'])
        rnn = keras.layers.RNN(cell, stateful=True)
        input_layer = keras.Input(batch_shape=(1, None, self.data_generator.number_of_inputs), ragged=True)
        output = rnn(input_layer)
        model = keras.models.Model(input_layer, output)
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return model

    def _test(self, train_inputs, train_outputs, test_inputs, test_outputs):
        self.model.reset_states()
        outputs = np.zeros(shape=(self.data_generator.number_of_test_samples, self.data_generator.number_of_outputs))
        state = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        forget = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        candidate = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        for i in range(self.data_generator.number_of_test_samples):
            if self.structure['forget_dnn_enable'] is True:
                outputs[i, :], state[i, :], forget[i, :], candidate[i, :] = self.model(train_inputs[i, :, :])
            else:
                outputs[i, :], state[i, :], candidate[i, :] = self.model(train_inputs[i, :, :])
        result_train = {'train_inputs': train_inputs,
                        'train_outputs': train_outputs,
                        'train_predicted_outputs': outputs,
                        'train_network_state': state,
                        'train_forget': forget,
                        'train_candidate': candidate}
        self.model.reset_states()
        outputs = np.zeros(shape=(self.data_generator.number_of_test_samples, self.data_generator.number_of_outputs))
        state = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        forget = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        candidate = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        for i in range(self.data_generator.number_of_test_samples):
            if self.structure['forget_dnn_enable'] is True:
                outputs[i, :], state[i, :], forget[i, :], candidate[i, :] = self.model(test_inputs[i, :, :])
            else:
                outputs[i, :], state[i, :], candidate[i, :] = self.model(test_inputs[i, :, :])
        result_test = {'test_inputs': test_inputs,
                       'test_outputs': test_outputs,
                       'test_predicted_outputs': outputs,
                       'test_network_state': state,
                       'test_forget': forget,
                       'test_candidate': candidate}
        return {**result_train, **result_test}
