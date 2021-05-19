from tensorflow import keras
import tensorflow as tf
import numpy as np

STATE_EXTENSION = 1


class NewGRU(keras.layers.Layer):
    def __init__(self, state_length, number_of_output, **kwargs):
        self.state_length = state_length
        self.number_of_output = number_of_output
        self.state_size = [tf.TensorShape(state_length)]
        super(NewGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1]
        self._build_node_1(input_shape)
        self._build_node_2(input_shape)
        self._build_node_3()

    def call(self, inputs, states):
        states = states[0]
        inputs = tf.concat([inputs, states], -1)
        node_1_output = self._call_node_1(inputs)
        node_2_output = self._call_node_2(inputs)
        part_2 = tf.subtract(node_1_output, states)
        part_2 = tf.math.multiply(node_2_output, part_2)
        new_states = tf.add(part_2, states)
        output = self._call_node_3(new_states)

        # tf.print('input = ', inputs, 'old NN state = ', states, 'new NN state = ', new_states)
        return output, new_states

    def _build_node_1(self, input_shape):
        self.node_1_input_kernel = self.add_weight(shape=(input_shape + self.state_length,
                                                          STATE_EXTENSION * self.state_length),
                                                   initializer="uniform",
                                                   name="node_1_input_kernel")

        self.node_1_input_bias = self.add_weight(shape=(STATE_EXTENSION * self.state_length,),
                                                 initializer="uniform",
                                                 name="node_1_input_bias")

        self.node_1_output_kernel = self.add_weight(shape=(STATE_EXTENSION * self.state_length, self.state_length),
                                                    initializer="uniform",
                                                    name="node_1_output_kernel")
        self.node_1_output_bias = self.add_weight(shape=(self.state_length,),
                                                  initializer="uniform",
                                                  name="node_1_output_bias")

    def _call_node_1(self, inputs):
        output_step_1 = tf.matmul(inputs, self.node_1_input_kernel)
        output_step_2 = tf.matmul(output_step_1, self.node_1_output_kernel+ self.node_1_output_bias)
        return output_step_2

    def _build_node_2(self, input_shape):
        self.node_2_input_kernel = self.add_weight(shape=(input_shape + self.state_length,
                                                          STATE_EXTENSION * self.state_length),
                                                   initializer="uniform",
                                                   name="node_2_input_kernel")
        self.node_2_input_bias = self.add_weight(shape=(STATE_EXTENSION * self.state_length,),
                                                 initializer="uniform",
                                                 name="node_2_input_bias")
        self.node_2_output_kernel = self.add_weight(shape=(STATE_EXTENSION * self.state_length, self.state_length),
                                                    initializer="uniform",
                                                    name="node_2_output_kernel")
        self.node_2_output_bias = self.add_weight(shape=(self.state_length,),
                                                  initializer="uniform",
                                                  name="node_2_output_bias")

    def _call_node_2(self, inputs):
        output_step_1 = tf.sigmoid(tf.matmul(inputs, self.node_2_input_kernel) + self.node_2_input_bias)
        output_step_2 = tf.sigmoid(tf.matmul(output_step_1, self.node_2_output_kernel) + self.node_2_output_bias)
        return output_step_2

    def _build_node_3(self):
        self.node_3_input_kernel = self.add_weight(shape=(self.state_length, 1),
                                                   initializer="uniform",
                                                   name="node_3_input_kernel")
        self.node_3_input_bias = self.add_weight(shape=(STATE_EXTENSION * self.state_length,),
                                                 initializer="uniform",
                                                 name="node_3_input_bias")
        self.node_3_hidden_kernel = self.add_weight(shape=(STATE_EXTENSION * self.state_length,
                                                           STATE_EXTENSION * self.state_length),
                                                    initializer="uniform",
                                                    name="node_3_hidden_kernel")
        self.node_3_hidden_bias = self.add_weight(shape=(STATE_EXTENSION * self.state_length,),
                                                  initializer="uniform",
                                                  name="node_3_hidden_bias")
        self.node_3_output_kernel = self.add_weight(shape=(STATE_EXTENSION * self.state_length, self.number_of_output),
                                                    initializer="uniform",
                                                    name="node_3_output_kernel")
        self.node_3_output_bias = self.add_weight(shape=(self.number_of_output,),
                                                  initializer="uniform",
                                                  name="node_3_output_bias")

    def _call_node_3(self, inputs):
        output_step_1 = tf.matmul(inputs, self.node_3_input_kernel)
        output_step_2 = tf.matmul(output_step_1, self.node_3_hidden_kernel) + self.node_3_hidden_bias
        output_step_3 = tf.matmul(output_step_2, self.node_3_output_kernel) + self.node_3_output_bias
        return output_step_3
