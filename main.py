from NewGRU import NewGRU
from tensorflow import keras
import tensorflow as tf
from SystemDefinition import LinearSystem
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetworkTeacher import TeacherLS, TeacherBenchMark


def create_neural_network(number_of_inputs,
                          number_of_outputs,
                          time_step,
                          neural_network_state_length):
    cell = NewGRU(neural_network_state_length, number_of_outputs)
    rnn = keras.layers.RNN(cell)
    input_1 = keras.Input((time_step, number_of_inputs))
    output = rnn(input_1)
    model = keras.models.Model(input_1, output)
    model.compile(optimizer="adam", loss="mse", metrics='msle')
    model.summary()
    return model


def create_linear_system():
    state_matrix = np.array([[0.7, 0.8], [0, 0.1]])
    input_matrix = np.array([[-1], [0.1]])
    output_matrix = np.array([[1, 0]])
    system = LinearSystem(state_matrix=state_matrix,
                          input_matrix=input_matrix,
                          output_matrix=output_matrix,
                          process_noise=True,
                          output_noise=True)
    init_state = np.array([[0], [0]])
    return system, init_state


if __name__ == '__main__':
    state_length = 50
    number_of_all_input_batch = 4000
    time_step = 32
    number_of_system_sample = number_of_all_input_batch + time_step
    '''
    inputs shape = (number_of_all_input_batch, time_step, number_of_system_inputs)
    output shape = (number_of_all_input_batch. number_of_system_outputs)
    '''
    linear_system, initial_state = create_linear_system()
    teacher = TeacherLS(linear_system, number_of_system_sample, initial_state, time_step)
    inputs, outputs = teacher.get_data()

    # teacher = TeacherBenchMark(time_step)
    # inputs, outputs = teacher.get_data()
    print(np.shape(inputs))
    print(np.shape(outputs))
    train_inputs = inputs[0: int(0.5*number_of_all_input_batch), :, :]
    train_outputs = outputs[0: int(0.5*number_of_all_input_batch), :]
    val_inputs = inputs[int(0.5*number_of_all_input_batch): int(0.8*number_of_all_input_batch), :, :]
    val_outputs = outputs[int(0.5*number_of_all_input_batch): int(0.8*number_of_all_input_batch), :]
    predict_inputs = inputs[int(0.9*number_of_all_input_batch): number_of_all_input_batch, :, :]
    real_outputs_for_predict = outputs[int(0.9*number_of_all_input_batch): number_of_all_input_batch, :]

    number_of_system_inputs = np.shape(inputs)[2]
    number_of_system_outputs = np.shape(outputs)[1]
    neural_network = create_neural_network(number_of_inputs=number_of_system_inputs,
                                           number_of_outputs=number_of_system_outputs,
                                           time_step=time_step,
                                           neural_network_state_length=state_length)
    predict_before_fit = neural_network.predict(predict_inputs)
    neural_network.fit(train_inputs,
                       train_outputs,
                       epochs=10,
                       validation_data=(val_inputs, val_outputs))
    predict_output = neural_network.predict(predict_inputs)
    plt.plot(predict_before_fit, label='predict output before fit')
    plt.plot(real_outputs_for_predict, label='real output')
    plt.plot(predict_output, label='predict output')
    plt.legend()
    plt.show()
