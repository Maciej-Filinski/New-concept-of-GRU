import numpy as np


class LinearSystem:
    """ Definition of the linear dynamical discrete system"""
    def __init__(self, state_matrix: np.ndarray, input_matrix: np.ndarray, output_matrix: np.ndarray):
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

    def linear_system_response(self, input_sequence: np.ndarray, initial_state: np.ndarray):
        """ Implementation of the linear dynamic discrete system in state space.

        The function calculates the output of the system according to equations:
        x_{k+1} = Ax_{k} + Bu_{k}
        y_{k} = Cx_{k},
        where A is the state matrix, B is input matrix and C is output matrix.

        :param input_sequence: array of u_{k} for k = 0, 1, 2, ... N - 1.
        :type input_sequence: np.ndarray
        :param initial_state: initial state of system x_{0}.
        :type initial_state: np.ndarray
        :return: output sequence of the linear system.
        :rtype: np.ndarray
        """
        y = []
        x = [initial_state]
        for k in range(input_sequence.shape[1]):
            x.append(np.dot(self.A, x[:, k]) + np.dot(self.C, input_sequence[:, k]))
            y.append(np.dot(self.C, x[:, k]))
        return y


