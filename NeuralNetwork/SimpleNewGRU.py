from tensorflow import keras
import tensorflow as tf


class SimpleNewGRU(keras.layers.Layer):
    def __init__(self, neural_network_state_length, number_of_outputs, **kwargs):
        self.neural_network_state_length = neural_network_state_length
        self.number_of_outputs = number_of_outputs
        self.state_size = [tf.TensorShape(neural_network_state_length)]
        super(SimpleNewGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1]
        self._build_candidate_dnn(input_shape)
        self._build_forget_dnn(input_shape)
        self._build_output_dnn()

    def call(self, inputs, states):
        states = states[0]
        inputs = tf.concat([inputs, states], -1)
        candidate = self._call_candidate_dnn(inputs)
        forget_factors = self._call_forget_dnn(inputs)
        state_update = tf.subtract(candidate, states)
        state_update = tf.math.multiply(state_update, forget_factors)
        new_states = tf.add(state_update, states)
        output = self._call_output_dnn(new_states)
        return output, new_states

    def _build_candidate_dnn(self, input_shape):
        self.cdnn_input_kernel = self.add_weight(shape=(input_shape + self.neural_network_state_length,
                                                        self.neural_network_state_length),
                                                 initializer="uniform",
                                                 name="node_1_input_kernel")

        self.cdnn_input_bias = self.add_weight(shape=(self.neural_network_state_length,),
                                               initializer="uniform",
                                               name="node_1_input_bias")

    def _call_candidate_dnn(self, inputs):
        output = tf.matmul(inputs, self.cdnn_input_kernel) + self.cdnn_input_bias
        return output

    def _build_forget_dnn(self, input_shape):
        self.fdnn_input_kernel = self.add_weight(shape=(input_shape + self.neural_network_state_length,
                                                        self.neural_network_state_length),
                                                 initializer="uniform",
                                                 name="node_2_input_kernel")
        self.fdnn_input_bias = self.add_weight(shape=(self.neural_network_state_length,),
                                               initializer="uniform",
                                               name="node_2_input_bias")

    def _call_forget_dnn(self, inputs):
        output = tf.sigmoid(tf.matmul(inputs, self.fdnn_input_kernel) + self.fdnn_input_bias)
        return output

    def _build_output_dnn(self):
        self.odnn_input_kernel = self.add_weight(shape=(self.neural_network_state_length, self.number_of_outputs),
                                                 initializer="uniform",
                                                 name="node_3_input_kernel")
        self.odnn_input_bias = self.add_weight(shape=(self.number_of_outputs,),
                                               initializer="uniform",
                                               name="node_3_input_bias")

    def _call_output_dnn(self, inputs):
        output_step_1 = tf.matmul(inputs, self.odnn_input_kernel) + self.odnn_input_bias
        return output_step_1
