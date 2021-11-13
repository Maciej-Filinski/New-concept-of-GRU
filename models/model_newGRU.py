import tensorflow as tf
from tensorflow import keras


def build_new_gru_for_train(structure, layer, number_of_inputs, number_of_outputs):
    cell = layer(state_length=structure['state_length'],
                 number_of_outputs=number_of_outputs,
                 candidate_dnn_structure=structure['candidate_dnn'],
                 forget_dnn_structure=structure['forget_dnn'],
                 output_dnn_structure=structure['output_dnn'],
                 forget_dnn_enable=structure['forget_dnn_enable'],
                 output_dnn_enable=structure['output_dnn_enable'],
                 return_full_output=False)
    rnn = keras.layers.RNN(cell, stateful=True)
    input_layer = keras.Input(batch_shape=(1, None, number_of_inputs), ragged=True)
    output = rnn(input_layer)
    model = keras.models.Model(input_layer, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def build_new_gru_for_test(structure, layer, number_of_inputs, number_of_outputs):
    cell = layer(state_length=structure['state_length'],
                 number_of_outputs=number_of_outputs,
                 candidate_dnn_structure=structure['candidate_dnn'],
                 forget_dnn_structure=structure['forget_dnn'],
                 output_dnn_structure=structure['output_dnn'],
                 forget_dnn_enable=structure['forget_dnn_enable'],
                 output_dnn_enable=structure['output_dnn_enable'],
                 return_full_output=True)
    rnn = keras.layers.RNN(cell, stateful=True)
    input_layer = keras.Input(batch_shape=(1, None, number_of_inputs), ragged=True)
    output = rnn(input_layer)
    model = keras.models.Model(input_layer, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
