import numpy as np
import datetime
from neural_network import NewGRULinear
import tensorflow as tf
from tensorflow import keras
import os
import json
from data_generator import ToyProblem, ToyProblemV2, ToyProblemV3
import matplotlib.pyplot as plt
import random

generator = ToyProblemV2(number_of_train_samples=100, number_of_test_samples=100, dataset_name='toy_problem_v2')
data = generator.load_data()
structure_path = os.path.join(os.path.abspath('simulation/neural_network_structure'),
                              'structure_2.json')
with open(structure_path, 'r') as file:
    structure = json.load(file)
cell = NewGRULinear(state_length=structure['state_length'],
                    number_of_outputs=1,
                    candidate_dnn_structure=structure['candidate_dnn'],
                    forget_dnn_structure=structure['forget_dnn'],
                    output_dnn_structure=structure['output_dnn'],
                    forget_dnn_enable=structure['forget_dnn_enable'],
                    output_dnn_enable=structure['output_dnn_enable'],
                    return_full_output=structure['return_full_output'])
rnn = keras.layers.RNN(cell, stateful=True)
input_layer = keras.Input(batch_shape=(1, None, 1), ragged=True)
output = rnn(input_layer)
model = keras.models.Model(input_layer, output)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()
# delta = 0.05
# weights = model.get_weights()
# weights[0] = np.concatenate([generator.system.input_matrix + delta, generator.system.state_matrix], axis=-1).transpose()
# weights[1] = np.zeros((2, )) + delta
# weights[2] = np.reshape(generator.system.output_matrix - delta, newshape=(2, 1))
# weights[3] = np.zeros(1,) - delta
# model.set_weights(weights)
print(model.get_weights())

number_of_epochs = 50
batch_size = 20
indices_set = list(range(generator.number_of_train_samples))
random.shuffle(indices_set)
number_of_batch = generator.number_of_train_samples // batch_size
execution_time = np.zeros(shape=(number_of_epochs, number_of_batch))
loss_function = np.zeros(shape=(number_of_epochs, number_of_batch))
x_train = data['train']['inputs']
y_train = data['train']['outputs']
for epoch in range(number_of_epochs):
    for step in range(number_of_batch):
        t_start = datetime.datetime.now()
        batch_shuffle_set = indices_set[step * batch_size: (step + 1) * batch_size]

        loss_function = tf.keras.losses.MeanSquaredError()

        max_value = np.max(batch_shuffle_set)
        model.reset_states()
        y_predicted = []
        with tf.GradientTape() as tape:
            for i in range(max_value + 1):
                y = model(x_train[i, :, :])
                y_predicted.append(y)
            y_predicted = tf.convert_to_tensor(y_predicted)
            y_1 = []
            y_2 = []
            for i in batch_shuffle_set:
                y_1.append(y_train[i, :])
                y_2.append(y_predicted[i, :])
            y_1 = tf.convert_to_tensor(y_1)
            y_2 = tf.convert_to_tensor(y_2)
            loss = loss_function(y_1, y_2)
        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

        t_end = datetime.datetime.now()
        execution_time[epoch, step] = (t_end - t_start).total_seconds()
        print(f'epoch:{epoch + 1:3}/{number_of_epochs:3}, step:{step + 1:3}/{number_of_batch:3},',
              f'step execution time:{execution_time[epoch, step]:.2f}s,',
              f'loss:{loss:.5f}')


model.reset_states()
print(model.get_weights())
predicted_output = np.zeros(shape=(generator.number_of_test_samples, 1))
for i in range(generator.number_of_test_samples):
    predicted_output[i, :] = model(data['test']['inputs'][i, :])
plt.plot(data['test']['outputs'], label='true outputs')
plt.plot(predicted_output, label='predicted outputs')
plt.legend()
plt.show()