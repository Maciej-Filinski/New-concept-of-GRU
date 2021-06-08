import tensorflow as tf
import matplotlib.pyplot as plt


def print_real_system_state(state):
    fig, ax = plt.subplots(2)
    ax[0].plot(state[0, :])
    ax[1].plot(state[1, :])


def print_neural_network_state():
    pass


def print_output(true_output, predicted_output):
    number_of_plots = true_output.shape[1]
    fig, ax = plt.subplots(nrows=number_of_plots)
    for i in range(number_of_plots):
        ax[i].plot(true_output, label='true output', linewidth=4)
        ax[i].plot(predicted_output, label='predicted output')
        ax[i].ylabel('y_n')
    plt.xlabel('n')
    plt.grid()
    plt.title('Trained on test_data with process and output noise (input = sin(2kpi/10) + sin(2kpi/25))')
    plt.legend()
    plt.show()
