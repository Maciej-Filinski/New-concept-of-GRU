import numpy as np

from neural_network import NewGRULinear
import tensorflow as tf
from tensorflow import keras
import os
import json
from data_generator import ToyProblem, ToyProblemV2, ToyProblemV3
import matplotlib.pyplot as plt

generator = ToyProblemV3(number_of_train_samples=100, number_of_test_samples=100, dataset_name='toy_problem_v3')
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

weights = model.get_weights()
weights[0] = np.concatenate([generator.system.input_matrix, generator.system.state_matrix], axis=-1).transpose()
weights[1] = np.zeros((2, ))
weights[2] = np.reshape(generator.system.output_matrix, newshape=(2, 1))
weights[3] = np.zeros(1,)
model.set_weights(weights)
weights = model.get_weights()
predicted_output = np.zeros(shape=(generator.number_of_test_samples, 1))
for i in range(generator.number_of_test_samples):
    predicted_output[i, :] = model(data['train']['inputs'][i, :])
print(weights)
plt.plot(data['train']['outputs'], label='true outputs')
plt.plot(predicted_output, label='predicted outputs')
plt.legend()
plt.show()