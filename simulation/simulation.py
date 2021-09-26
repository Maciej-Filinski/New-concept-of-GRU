from data_generator import DataGenerator
from simulation import Result
from models import AbstractModel
import tensorflow as tf
import random
import numpy as np
import datetime
# TODO: Rebuild model class add rest function


class Simulation:
    def __init__(self,
                 data_generator: DataGenerator,
                 model_for_train: AbstractModel,
                 model_for_test: AbstractModel,
                 simulation_number: int = 1):

        self.data_generator = data_generator
        self.model = model_for_train
        self.model_for_test = model_for_test
        self.structure = self.model.structure
        self.result = Result(problem_name=self.data_generator.__name__,
                             structure=self.structure,
                             number_of_train_sample=self.data_generator.number_of_train_samples,
                             simulation_number=simulation_number)
        self.number_of_epochs = None
        self.batch_size_list = None
        self.shuffle = None

    def set_param_for_train(self,
                            number_of_epochs: int,
                            batch_size_list: int,
                            shuffle: bool = False):
        self.number_of_epochs = number_of_epochs
        self.batch_size_list = batch_size_list
        self.shuffle = shuffle

    def run(self):
        data = self.data_generator.load_data()
        for batch_size in self.batch_size_list:
            indices_set = list(range(self.data_generator.number_of_train_samples))
            if self.shuffle is True:
                random.shuffle(indices_set)
            number_of_batch = self.data_generator.number_of_train_samples // batch_size
            execution_time = np.zeros(shape=(self.number_of_epochs, number_of_batch))
            loss_function = np.zeros(shape=(self.number_of_epochs, number_of_batch))
            for epoch in range(self.number_of_epochs):

                for step in range(number_of_batch):
                    t_start = datetime.datetime.now()
                    batch_shuffle_set = indices_set[step * batch_size: (step + 1) * batch_size]

                    loss = self._run_one_step(x_train=data['train']['inputs'], y_train=data['train']['outputs'],
                                              batch_shuffle_set=batch_shuffle_set)

                    t_end = datetime.datetime.now()
                    execution_time[epoch, step] = (t_end - t_start).total_seconds()
                    loss_function[epoch, step] = loss
                    print(f'epoch:{epoch + 1:3}/{self.number_of_epochs:3}, step:{step + 1:3}/{number_of_batch:3},',
                          f'step execution time:{execution_time[epoch, step]:.2f}s,',
                          f'loss:{loss:.8f}')

            train_info = {'loss_function': loss_function,
                          'execution_time': execution_time}
            self.model.save_weights(self.result.model_path)
            weights = self.model.get_weights()
            self.structure['return_full_output'] = True
            self.model = self.model.build_model(self.structure)
            self.model.set_weights(weights)
            test_info = self._test(train_inputs=data['train']['inputs'][:self.data_generator.number_of_test_samples, :, :],
                                   train_outputs=data['train']['outputs'][:self.data_generator.number_of_test_samples, :],
                                   test_inputs=data['test']['inputs'],
                                   test_outputs=data['test']['outputs'])
            self.result.data = {**train_info, **test_info}
            self.result.save()

    def _run_one_step(self,
                      x_train: np.ndarray,
                      y_train: np.ndarray,
                      batch_shuffle_set: list):
        loss_function = tf.keras.losses.MeanSquaredError()

        max_value = np.max(batch_shuffle_set)
        self.model.reset_states()
        y_predicted = []
        with tf.GradientTape() as tape:
            if self.structure['return_full_output'] is True:
                for i in range(max_value + 1):
                    y, *rest = self.model.predict_one_step(x_train[i, :, :])
                    y_predicted.append(y)
            else:
                for i in range(max_value + 1):
                    y = self.model.predict_one_step(x_train[i, :, :])
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
        grads = tape.gradient(loss, self.model.model.trainable_weights)
        self.model.model.optimizer.apply_gradients(zip(grads, self.model.model.trainable_weights))
        return loss

    def _test(self, train_inputs, train_outputs, test_inputs, test_outputs):
        self.model.reset_states()
        outputs = np.zeros(shape=(self.data_generator.number_of_test_samples, self.data_generator.number_of_outputs))
        state = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        forget = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        candidate = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        for i in range(self.data_generator.number_of_test_samples):
            if self.structure['forget_dnn_enable'] is True:
                outputs[i, :], state[i, :], forget[i, :], candidate[i, :] = self.model.model(train_inputs[i, :, :])
            else:
                outputs[i, :], state[i, :], candidate[i, :] = self.model.model(train_inputs[i, :, :])
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
                outputs[i, :], state[i, :], forget[i, :], candidate[i, :] = self.model.model(test_inputs[i, :, :])
            else:
                outputs[i, :], state[i, :], candidate[i, :] = self.model.model(test_inputs[i, :, :])
        result_test = {'test_inputs': test_inputs,
                       'test_outputs': test_outputs,
                       'test_predicted_outputs': outputs,
                       'test_network_state': state,
                       'test_forget': forget,
                       'test_candidate': candidate}
        return {**result_train, **result_test}
