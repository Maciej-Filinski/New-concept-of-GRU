import numpy as np
import matplotlib.pyplot as plt


class NonlinearSystemBenchmark:
    def __init__(self):
        pass

    def response(self, input_sequence):
        # see andreas lindholm's work "A flexible state-space model for learning nonlinear dynamical systems"

        # get length of input
        k_max = input_sequence.shape[-1]

        # allocation
        x = np.zeros([2, k_max + 1])
        y = np.zeros([1, k_max + 1])

        # run over all time steps
        for k in range(k_max):
            # state 1
            x[0, k + 1] = (x[0, k] / (1 + x[0, k] ** 2) + 1) * np.sin(x[1, k])
            # state 2
            term1 = x[1, k] * np.cos(x[1, k])
            term2 = x[0, k] * np.exp(-1 / 8 * (x[0, k] ** 2 + x[1, k] ** 2))
            term3 = input_sequence[0, k] ** 3 / (1 + input_sequence[0, k] ** 2 + 0.5 * np.cos(x[0, k] + x[1, k]))
            x[1, k + 1] = term1 + term2 + term3
            # output
            term1 = x[0, k] / (1 + 0.5 * np.sin(x[1, k]))
            term2 = x[1, k] / (1 + 0.5 * np.sin(x[0, k]))
            y[0, k] = term1 + term2
        term1 = x[0, -1] / (1 + 0.5 * np.sin(x[1, -1]))
        term2 = x[1, -1] / (1 + 0.5 * np.sin(x[0, -1]))
        y[0, -1] = term1 + term2
        return y


if __name__ == "__main__":
    ns = NonlinearSystemBenchmark()
    input_sequence = np.array(list(range(100))).reshape((1, 100))
    print(np.shape(input_sequence))
    input_sequence = np.sin(2 * np.pi * input_sequence / 10) + np.sin(2 * np.pi * input_sequence / 25)
    output_sequence = ns.response(input_sequence)
    print(np.shape(output_sequence))
    plt.plot(output_sequence.transpose())
    plt.show()
