import numpy as np


class ToyProblemSystem:
    state_matrix = np.array([[0.7555, 0.25], [-0.1991, 0]])
    input_matrix = np.array([[-0.5], [0]])
    output_matrix = np.array([0.6993, -0.4427])

    def __init__(self, noise=False, noise_variance=1):
        self.noise = noise
        self.noise_variance = noise_variance

    def response(self, inputs: np.ndarray) -> dict:
        """
        :param inputs:
        :return dict:
            {'inputs': [u_0, u_1, ..., u_N, 'outputs': [y_1, y_2, ..., y_N+1}, 'states': [x_1, x_2, ..., x_N+1]}
        """
        number_of_samples = inputs.shape[0]
        states = np.zeros(shape=(number_of_samples + 1, 2))
        outputs = np.zeros(shape=(number_of_samples + 1, 1))
        for i in range(1, number_of_samples + 1):
            if inputs[i - 1, :] > 0:
                states[i, :] = self.state_matrix @ states[i - 1, :] + self.input_matrix @ np.sqrt(inputs[i - 1, :])
            else:
                states[i, :] = self.state_matrix @ states[i - 1, :] + self.input_matrix @ inputs[i - 1, :]
        states = states[1::, :]
        for i in range(number_of_samples):
            outputs[i, :] = self.output_matrix @ states[i, :] + 5 * np.sin(self.output_matrix @ states[i, :])
        # TODO: implement output noise.
        return {'inputs': inputs, 'outputs': outputs, 'states': states}


class ToyProblemSystemV2(ToyProblemSystem):
    state_matrix = np.array([[1.3652, -0.8259], [1, 0]])
    input_matrix = np.array([[1], [0]])
    output_matrix = np.array([-0.3497, -0.0441])

    def response(self, inputs: np.ndarray) -> dict:
        """
        :param inputs:
        :return dict:
            {'inputs': [u_0, u_1, ..., u_N, 'outputs': [y_1, y_2, ..., y_N+1}, 'states': [x_1, x_2, ..., x_N+1]}
        """
        number_of_samples = inputs.shape[0]
        states = np.zeros(shape=(number_of_samples + 1, 2))
        outputs = np.zeros(shape=(number_of_samples + 1, 1))
        for i in range(1, number_of_samples + 1):
            states[i, :] = self.state_matrix @ states[i - 1, :] + self.input_matrix @ inputs[i - 1, :]
        states = states[1::, :]
        for i in range(number_of_samples):
            outputs[i, :] = self.output_matrix @ states[i, :]
        # TODO: implement output noise.
        return {'inputs': inputs, 'outputs': outputs, 'states': states}


class ToyProblemSystemV3(ToyProblemSystem):
    state_matrix = np.array([[1.3652, -0.4659], [1, 0]])
    input_matrix = np.array([[1], [0]])
    output_matrix = np.array([-0.3497, -0.0441])

    def response(self, inputs: np.ndarray) -> dict:
        """
        :param inputs:
        :return dict:
            {'inputs': [u_0, u_1, ..., u_N, 'outputs': [y_1, y_2, ..., y_N+1}, 'states': [x_1, x_2, ..., x_N+1]}
        """
        number_of_samples = inputs.shape[0]
        states = np.zeros(shape=(number_of_samples + 1, 2))
        outputs = np.zeros(shape=(number_of_samples + 1, 1))
        for i in range(1, number_of_samples + 1):
            states[i, :] = self.state_matrix @ states[i - 1, :] + self.input_matrix @ inputs[i - 1, :]
        states = states[1::, :]
        for i in range(number_of_samples):
            outputs[i, :] = self.output_matrix @ states[i, :]
        # TODO: implement output noise.
        return {'inputs': inputs, 'outputs': outputs, 'states': states}
