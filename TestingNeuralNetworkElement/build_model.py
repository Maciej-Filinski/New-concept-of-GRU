from NeuralNetwork import NewGRU
import tensorflow as tf
from tensorflow import keras
import constants as c
import structure_1 as s1
import structure_2 as s2
import structure_3 as s3


def build_model(ss_vector_length, number_of_inputs, batch_size, stateful=True, full_output=False):
    cell = NewGRU(state_length=c.NEURAL_NETWORK_STATE_LENGTH,
                  number_of_outputs=ss_vector_length,
                  candidate_dnn_structure=c.CANDIDATE_DNN_STRUCTURE,
                  forget_dnn_structure=c.FORGET_DNN_STRUCTURE,
                  output_dnn_structure=c.OUTPUT_DNN_STRUCTURE,
                  forget_dnn_enable=True,
                  output_dnn_enable=False,
                  full_output=full_output)
    rnn = keras.layers.RNN(cell, stateful=stateful)
    input_layer = keras.Input(batch_shape=(batch_size, None, number_of_inputs), ragged=True)
    output = rnn(input_layer)
    model = keras.models.Model(input_layer, output)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def build_model_input_state(ss_vector_length, number_of_inputs, batch_size, struct, stateful=True, full_output=False):
    if struct == 1:
        cell = NewGRU(state_length=s1.NEURAL_NETWORK_STATE_LENGTH,
                      number_of_outputs=ss_vector_length,
                      candidate_dnn_structure=s1.CANDIDATE_DNN_STRUCTURE,
                      forget_dnn_structure=s1.FORGET_DNN_STRUCTURE,
                      output_dnn_structure=s1.OUTPUT_DNN_STRUCTURE,
                      forget_dnn_enable=True,
                      output_dnn_enable=False,
                      full_output=full_output)
    elif struct == 2:
        cell = NewGRU(state_length=s2.NEURAL_NETWORK_STATE_LENGTH,
                      number_of_outputs=ss_vector_length,
                      candidate_dnn_structure=s2.CANDIDATE_DNN_STRUCTURE,
                      forget_dnn_structure=s2.FORGET_DNN_STRUCTURE,
                      output_dnn_structure=s2.OUTPUT_DNN_STRUCTURE,
                      forget_dnn_enable=True,
                      output_dnn_enable=False,
                      full_output=full_output)
    else:
        cell = NewGRU(state_length=s3.NEURAL_NETWORK_STATE_LENGTH,
                      number_of_outputs=ss_vector_length,
                      candidate_dnn_structure=s3.CANDIDATE_DNN_STRUCTURE,
                      forget_dnn_structure=s3.FORGET_DNN_STRUCTURE,
                      output_dnn_structure=s3.OUTPUT_DNN_STRUCTURE,
                      forget_dnn_enable=True,
                      output_dnn_enable=False,
                      full_output=full_output)
    rnn = keras.layers.RNN(cell, stateful=stateful)
    input_layer = keras.Input(batch_shape=(batch_size, None, number_of_inputs), ragged=True)
    output = rnn(input_layer)
    model = keras.models.Model(input_layer, output)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def build_model_input_output(number_of_outputs, number_of_inputs, batch_size, struct, stateful=True, full_output=False):
    if struct == 1:
        cell = NewGRU(state_length=s1.NEURAL_NETWORK_STATE_LENGTH,
                      number_of_outputs=number_of_outputs,
                      candidate_dnn_structure=s1.CANDIDATE_DNN_STRUCTURE,
                      forget_dnn_structure=s1.FORGET_DNN_STRUCTURE,
                      output_dnn_structure=s1.OUTPUT_DNN_STRUCTURE,
                      forget_dnn_enable=True,
                      output_dnn_enable=True,
                      full_output=full_output)
    elif struct == 2:
        cell = NewGRU(state_length=s2.NEURAL_NETWORK_STATE_LENGTH,
                      number_of_outputs=number_of_outputs,
                      candidate_dnn_structure=s2.CANDIDATE_DNN_STRUCTURE,
                      forget_dnn_structure=s2.FORGET_DNN_STRUCTURE,
                      output_dnn_structure=s2.OUTPUT_DNN_STRUCTURE,
                      forget_dnn_enable=True,
                      output_dnn_enable=True,
                      full_output=full_output)
    else:
        cell = NewGRU(state_length=s3.NEURAL_NETWORK_STATE_LENGTH,
                      number_of_outputs=number_of_outputs,
                      candidate_dnn_structure=s3.CANDIDATE_DNN_STRUCTURE,
                      forget_dnn_structure=s3.FORGET_DNN_STRUCTURE,
                      output_dnn_structure=s3.OUTPUT_DNN_STRUCTURE,
                      forget_dnn_enable=True,
                      output_dnn_enable=True,
                      full_output=full_output)
    rnn = keras.layers.RNN(cell, stateful=stateful)
    input_layer = keras.Input(batch_shape=(batch_size, None, number_of_inputs), ragged=True)
    output = rnn(input_layer)
    model = keras.models.Model(input_layer, output)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
