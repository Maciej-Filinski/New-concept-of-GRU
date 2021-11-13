from data_generator import DataGenerator
from simulation import Result
import tensorflow as tf
import random
import numpy as np
import datetime


class Simulation:
    def __init__(self,
                 data_generator: DataGenerator,
                 model_for_train: tf.keras.models.Model,
                 model_for_test: tf.keras.models.Model,
                 structure: dict,
                 simulation_number: int = 1):
        self.data_generator = data_generator
        self.data = self.data_generator.load_data()
        self.model = model_for_train
        self.model_for_test = model_for_test
        self.structure = structure
        self.result = Result(problem_name=self.data_generator.__name__,
                             structure=self.structure,
                             number_of_train_sample=self.data_generator.number_of_train_samples,
                             simulation_number=simulation_number)
        self.epochs_list = None
        self.batch_size_list = None
        self.shuffle = None

    def set_param_for_train(self,
                            epochs_list: list,
                            batch_size_list: list,
                            shuffle: bool = False):
        self.epochs_list = epochs_list
        self.batch_size_list = batch_size_list
        self.shuffle = shuffle

    def run(self):

        number_of_test_samples = self.data_generator.number_of_test_samples
        number_of_train_samples = self.data_generator.number_of_train_samples
        execution_time = []
        loss_function_train = []
        loss_function_test = []
        for number_of_epochs, batch_size in zip(self.epochs_list, self.batch_size_list):
            indices_set = list(range(number_of_train_samples))
            if self.shuffle is True:
                random.shuffle(indices_set)
            number_of_batch = number_of_train_samples // batch_size
            for epoch in range(number_of_epochs):
                start_epoch_time = datetime.datetime.now()
                for step in range(number_of_batch):
                    start_step_time = datetime.datetime.now()
                    batch_shuffle_set = indices_set[step * batch_size: (step + 1) * batch_size]

                    loss = self._run_one_step(x_train=self.data['train']['inputs'], y_train=self.data['train']['outputs'],
                                              batch_shuffle_set=batch_shuffle_set)

                    end_step_time = datetime.datetime.now()
                    print(f'epoch:{epoch + 1:3}/{number_of_epochs:3}, step:{step + 1:3}/{number_of_batch:3},',
                          f'step execution time:{(end_step_time - start_step_time).total_seconds():.2f}s,',
                          f'loss:{loss:.8f}')

                end_epoch_time = datetime.datetime.now()
                loss_for_all_data = self._calculate_loss(train_inputs=self.data['train']['inputs'],
                                                         train_outputs=self.data['train']['outputs'],
                                                         test_inputs=self.data['test']['inputs'],
                                                         test_outputs=self.data['test']['outputs'])
                loss_function_train.append(loss_for_all_data[0])
                loss_function_test.append(loss_for_all_data[1])
                execution_time.append((end_epoch_time - start_epoch_time).total_seconds())
        train_info = {'loss_function_train': loss_function_train,
                      'loss_function_test': loss_function_test,
                      'execution_time': execution_time}

        weights = self.model.get_weights()
        self.model_for_test.set_weights(weights)
        test_info = self._test(train_inputs=self.data['train']['inputs'][:number_of_test_samples, :, :],
                               train_outputs=self.data['train']['outputs'][:number_of_test_samples, :],
                               test_inputs=self.data['test']['inputs'],
                               test_outputs=self.data['test']['outputs'])
        self.result.data = {**train_info, **test_info}
        self.result.save()
        self.model.save_weights(self.result.model_path)

    def _run_one_step(self,
                      x_train: np.ndarray,
                      y_train: np.ndarray,
                      batch_shuffle_set: list):

        max_value = np.max(batch_shuffle_set)
        self.model.reset_states()
        y_predicted = []
        with tf.GradientTape() as tape:
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
            loss = self.model.compiled_loss(y_1, y_2)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def _test(self, train_inputs, train_outputs, test_inputs, test_outputs):
        self.model_for_test.reset_states()
        outputs = np.zeros(shape=(self.data_generator.number_of_test_samples, self.data_generator.number_of_outputs))
        state = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        forget = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        candidate = np.zeros(shape=(self.data_generator.number_of_test_samples, self.structure['state_length']))
        for i in range(self.data_generator.number_of_test_samples):
            if self.structure['forget_dnn_enable'] is True:
                outputs[i, :], state[i, :], forget[i, :], candidate[i, :] = self.model_for_test(train_inputs[i, :, :])
            else:
                outputs[i, :], state[i, :], candidate[i, :] = self.model_for_test(train_inputs[i, :, :])
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
                outputs[i, :], state[i, :], forget[i, :], candidate[i, :] = self.model_for_test(test_inputs[i, :, :])
            else:
                outputs[i, :], state[i, :], candidate[i, :] = self.model_for_test(test_inputs[i, :, :])
        result_test = {'test_inputs': test_inputs,
                       'test_outputs': test_outputs,
                       'test_predicted_outputs': outputs,
                       'test_network_state': state,
                       'test_forget': forget,
                       'test_candidate': candidate}
        return {**result_train, **result_test}

    def _calculate_loss(self, train_inputs, train_outputs, test_inputs, test_outputs):
        self.model.reset_states()
        predicted_train_outputs = np.zeros(shape=(self.data_generator.number_of_train_samples,
                                                  self.data_generator.number_of_outputs))
        predicted_test_outputs = np.zeros(shape=(self.data_generator.number_of_test_samples,
                                                 self.data_generator.number_of_outputs))
        for i in range(self.data_generator.number_of_train_samples):
            predicted_train_outputs[i, :] = self.model(train_inputs[i, :, :])
        for i in range(self.data_generator.number_of_test_samples):
            predicted_test_outputs[i, :] = self.model(test_inputs[i, :, :])

        train_loss = self.model.compiled_loss(tf.convert_to_tensor(predicted_train_outputs), tf.convert_to_tensor(train_outputs))
        test_loss = self.model.compiled_loss(tf.convert_to_tensor(predicted_test_outputs), tf.convert_to_tensor(test_outputs))
        return train_loss, test_loss
