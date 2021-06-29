import numpy as np
import SystemDefinition.constants as c
import matplotlib.pyplot as plt


class LinearStateSpaceSystem:
    """ Definition of the linear dynamical discrete system"""
    def __init__(self, state_matrix: np.ndarray,
                 input_matrix: np.ndarray,
                 output_matrix: np.ndarray,
                 process_noise=False,
                 output_noise=False):
        """ Initiation of the linear dynamical discrete system.

        n - number of state variables
        p - number of inputs
        q - number of outputs

        :param state_matrix: array shape (n, n)
        :param input_matrix: array shape (n, p)
        :param output_matrix: array shape (q, n)
        """
        self.A = state_matrix
        self.B = input_matrix
        self.C = output_matrix
        self.n = state_matrix.shape[0]
        self.q = output_matrix.shape[0]
        self.p = input_matrix.shape[1]
        self.process_noise = process_noise
        self.output_noise = output_noise

        self.state_space = None

    def linear_system_response(self, input_sequence: np.ndarray, initial_state: np.ndarray):
        """ Implementation of the linear dynamic discrete system in state space.

        The function calculates the output of the system according to equations:
        x_{k+1} = Ax_{k} + Bu_{k}
        y_{k} = Cx_{k},
        where A is the state matrix, B is input matrix and C is output matrix.
        The function ignores the first output y_{1} because it's independent for known input.
        The function return two same length array i.e.
        [u_{1}, u_{2}, ..., u_{N-1}] -> length N
        [x_{1}, x_{2}, ..., x_{N}]   -> length N + 1
        [y_{1}, y_{2}, ..., y_{N}]   -> length N + 1

        :param input_sequence: array of u_{k} for k = 1, 2, ..., N - 1. shape=(N, p)
        :type input_sequence: np.ndarray
        :param initial_state: the initial state of the system x_{1}. shape=(n, N)
        :type initial_state: np.ndarray
        :return: the output sequence of the linear system. shape=(N, q)
        :rtype: np.ndarray
        """
        # TODO: Change shape of state matrix form (n, N) to (N, n)
        number_of_output_samples = input_sequence.shape[0]
        output_sequence = np.zeros((number_of_output_samples + 1, self.q))
        state_space_sequence = np.zeros((self.n, number_of_output_samples + 1))
        state_space_sequence[:, 0] = initial_state.reshape(initial_state.shape[0], )
        for k in range(number_of_output_samples):
            state_space_sequence[:, k + 1: k + 2] = self.A @ state_space_sequence[:, k: k + 1] + self.B @ input_sequence[k: k + 1, :]
            # state_space_sequence[:, k + 1: k + 2] += 0.5 * np.tanh(state_space_sequence[:, k: k + 1])
            if self.process_noise is True:
                state_space_sequence[:, k + 1: k + 2] += np.random.normal(c.EV_PROCESS_NOISE,
                                                                          c.VAR_PROCESS_NOISE,
                                                                          size=(self.n, 1))
            output_sequence[k, :] = self.C @ state_space_sequence[:, k: k + 1]
            # output_sequence[k, :] += 2 * state_space_sequence[1, k: k + 1] * state_space_sequence[0, k: k + 1]
            if self.output_noise is True:
                output_sequence[k, :] += np.random.normal(c.EV_OUTPUT_NOISE,
                                                          c.VAR_OUTPUT_NOISE,
                                                          size=(self.q, ))
        output_sequence[-1, :] = self.C @ state_space_sequence[:, -2: -1]
        self.state_space = state_space_sequence
        fig, ax = plt.subplots(2, num='True system state noisy')
        ax[0].plot(state_space_sequence[0, 200::])
        ax[1].plot(state_space_sequence[1, 200::])

        return output_sequence

    def return_state_space(self):
        return self.state_space
