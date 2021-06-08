import numpy as np
import matplotlib.pyplot as plt


class NewGRUFromScratch:
    def __init__(self, candidate_weight, candidate_bias, forget_weight, forget_bias, output_weight, output_bias):
        self.number_of_candidate_layer = len(candidate_weight)
        self.candidate_weight = candidate_weight
        self.candidate_bias = candidate_bias
        self.number_of_forget_layer = len(forget_weight)
        self.forget_weight = forget_weight
        self.forget_bias = forget_bias
        self.number_of_output_layer = len(output_weight)
        self.output_weight = output_weight
        self.output_bias = output_bias
        self.state_length = self.candidate_weight[self.number_of_candidate_layer]

    def _call_single_step(self, inputs, state):
        """

        :param inputs: single input shape=(1, number_of_inputs)
        :param state: single input shape=(1, state_length)
        :return:
        """
        candidate = np.concatenate((inputs, state), axis=-1)
        for i in range(self.number_of_candidate_layer):
            candidate = np.dot(candidate, self.candidate_weight[i]) + self.candidate_bias[i]

        forget = np.concatenate((inputs, state), axis=-1)
        for i in range(self.number_of_forget_layer):
            forget = np.dot(forget, self.forget_weight[i]) + self.forget_bias[i]

        candidate -= state
        delta_state = candidate * forget
        state += delta_state

        output = state
        for i in range(self.number_of_forget_layer):
            output = np.dot(output, self.forget_weight[i]) + self.forget_bias[i]

        return output, state

    def _call_all_recurrent_step(self, inputs):
        """

        :param inputs: all inputs to calculate one output shape=(time_step, number_of_inputs)
        :return:
        """
        inputs_length = np.shape(inputs)[0]
        state = np.zeros((1, self.state_length))
        for i in range(inputs_length):
            pass




if __name__ == '__main__':
    pass
