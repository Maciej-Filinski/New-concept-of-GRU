from NeuralNetwork import NewGRU, SimpleNewGRU
from tensorflow import keras
import tensorflow as tf
from SystemDefinition import LinearSystem
import matplotlib.pyplot as plt
import numpy as np
from DataPreprocessing.TeacherLS import TeacherLS
from Visualizations import print_output


def create_neural_network(number_of_inputs,
                          number_of_outputs,
                          time_step,
                          neural_network_state_length):
    cell = SimpleNewGRU(neural_network_state_length=neural_network_state_length, number_of_outputs=number_of_outputs)
    rnn = keras.layers.RNN(cell)
    input_1 = keras.Input((time_step, number_of_inputs))
    output = rnn(input_1)
    model = keras.models.Model(input_1, output)
    model.compile(optimizer="adam", loss="mse", metrics=tf.keras.metrics.RootMeanSquaredError())
    model.summary()
    return model


def create_linear_system():
    state_matrix = np.array([[0.7, 0.8], [0, 0.1]])
    input_matrix = np.array([[-1], [0.1]])
    output_matrix = np.array([[1, 1]])
    system = LinearSystem(state_matrix=state_matrix,
                          input_matrix=input_matrix,
                          output_matrix=output_matrix,
                          process_noise=False,
                          output_noise=True)
    init_state = np.array([[0], [0]])
    return system, init_state


if __name__ == '__main__':
    nn_state_length = 2
    number_of_all_input_batch = 4000
    time_step = 200
    number_of_system_sample = number_of_all_input_batch + time_step
    '''
    inputs shape = (number_of_all_input_batch, time_step, number_of_system_inputs)
    output shape = (number_of_all_input_batch. number_of_system_outputs)
    '''
    linear_system, initial_state = create_linear_system()
    teacher = TeacherLS(linear_system, number_of_system_sample, initial_state, time_step)
    inputs, outputs = teacher.get_data()
    print(f'max = {max(outputs)}, min = {min(outputs)}')
    # teacher = TeacherBenchMark(time_step)
    # inputs, outputs = teacher.get_data()
    print(np.shape(inputs))
    print(np.shape(outputs))
    train_inputs = inputs[0: int(0.5*number_of_all_input_batch), :, :]
    train_outputs = outputs[0: int(0.5*number_of_all_input_batch), :]
    val_inputs = inputs[int(0.5*number_of_all_input_batch): int(0.8*number_of_all_input_batch), :, :]
    val_outputs = outputs[int(0.5*number_of_all_input_batch): int(0.8*number_of_all_input_batch), :]
    predict_inputs = inputs[int(0.9*number_of_all_input_batch): number_of_all_input_batch, :, :]
    true_output = outputs[int(0.9 * number_of_all_input_batch): number_of_all_input_batch, :]

    teacher.system.output_noise = False
    teacher.system.process_noise = False
    predict_inputs, true_output = teacher.get_data_test()
    teacher.system.output_noise = True
    teacher.system.process_noise = True
    _, true_output_noisy = teacher.get_data_test()
    number_of_system_inputs = np.shape(inputs)[2]
    number_of_system_outputs = np.shape(outputs)[1]
    neural_network = create_neural_network(number_of_inputs=number_of_system_inputs,
                                           number_of_outputs=number_of_system_outputs,
                                           time_step=time_step,
                                           neural_network_state_length=nn_state_length)
    #weight = neural_network.get_weights()
    #weight[0] = np.array([[-1, 0.1], [0.7, 0], [0.8, 0.1]])
    #weight[4] = np.array([[1], [0]])
    #neural_network.set_weights(weight)
    predicted_output_before = neural_network.predict(predict_inputs)
    neural_network.fit(train_inputs,
                       train_outputs,
                       epochs=20,
                       validation_data=(val_inputs, val_outputs))
    predicted_output_train_data = neural_network(train_inputs)

    weights = neural_network.get_weights()
    print(weights)
    states = []
    outputs = []
    f = []
    c = []
    for input in predict_inputs:
        state = np.zeros(shape=(nn_state_length, ))
        forget_factors = np.zeros(shape=(nn_state_length, ))
        candidate = np.zeros(shape=(nn_state_length, ))
        for i in range(time_step):
            nn_input = np.concatenate([np.array(input[i]), state])
            candidate = np.matmul(nn_input, weights[0]) + weights[1]
            forget_factors = 1/(1 + np.exp(-(np.matmul(nn_input, weights[2]) + weights[3])))
            state_update = candidate - state
            state_update = np.multiply(state_update, forget_factors)
            state = state_update + state
        f.append(forget_factors)
        c.append(candidate)
        output = np.matmul(state, weights[4]) + weights[5]
        outputs.append(output)
        states.append(state)
    f = np.array(f)
    c = np.array(c)
    states = np.array(states)
    fig, ax = plt.subplots(3)
    ax[0].plot(states[:, 0])
    ax[1].plot(states[:, 1])
    ax[2].plot(states[:, 0] + states[:, 1])
    plt.title('neural network state')

    fig, ax = plt.subplots(2)
    ax[0].plot(f[:, 0])
    ax[1].plot(f[:, 1])
    plt.title('neural network f')

    fig, ax = plt.subplots(2)
    ax[0].plot(c[:, 0])
    ax[1].plot(c[:, 1])
    plt.title('neural network cand')

    plt.figure()
    plt.plot(train_outputs, label='true noisy output')
    plt.plot(predicted_output_train_data, label='predicted output')
    plt.xlabel('n')
    plt.ylabel('y_n')
    plt.title('Trained on data with process and output noise (random input sequence)')
    plt.legend()
    plt.grid()

    predicted_output = neural_network.predict(predict_inputs)
    plt.figure()
    plt.plot(true_output, label='true noisefree output', linewidth=4)
    plt.plot(predicted_output, label='predicted output')
    plt.plot(true_output_noisy, label='true noisy output')
    plt.xlabel('n')
    plt.ylabel('y_n')
    plt.grid()
    plt.title('Trained on data with process and output noise (input = sin(2kpi/10) + sin(2kpi/25))')
    plt.legend()
    plt.show()
