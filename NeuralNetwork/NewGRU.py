from tensorflow import keras
import tensorflow as tf
import numpy as np
import NeuralNetwork.constants as c


class NewGRU(keras.layers.Layer):
    def __init__(self, state_length, number_of_output, **kwargs):
        self.state_length = state_length
        self.number_of_output = number_of_output
        self.state_size = [tf.TensorShape(state_length)]
        super(NewGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1]
        self._build_candidate_dnn(input_shape)
        self._build_forget_dnn(input_shape)
        self._build_output_dnn()

    def call(self, inputs, states):
        states = states[0]
        inputs = tf.concat([inputs, states], -1)
        node_1_output = self._call_candidate_dnn(inputs)
        node_2_output = self._call_forget_dnn(inputs)
        part_2 = tf.subtract(node_1_output, states)
        part_2 = tf.math.multiply(node_2_output, part_2)
        new_states = tf.add(part_2, states)
        output = self._call_output_dnn(new_states)

        #tf.print('inputs =', inputs)
        #tf.print('state = ', states)
        #tf.print('new state = ', new_states)
        #tf.print('output = ', output)
        return output, new_states

    def _build_candidate_dnn(self, input_shape):
        self.candidate_dnn_input_kernel = self.add_weight(shape=(input_shape + self.state_length,
                                                                 c.STATE_EXTENSION * self.state_length),
                                                          initializer="uniform",
                                                          name="node_1_input_kernel")
        self.candidate_dnn_input_bias = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,),
                                                        initializer="uniform",
                                                        name="node_1_input_bias")

        self.candidate_dnn_hidden_kernel_1 = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,
                                                                    c.STATE_EXTENSION**2 * self.state_length),
                                                             initializer="uniform",
                                                             name="node_1_input_kernel")
        self.candidate_dnn_hidden_bias_1 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,),
                                                           initializer="uniform",
                                                           name="node_1_input_bias")

        self.candidate_dnn_hidden_kernel_2 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,
                                                                    c.STATE_EXTENSION**3 * self.state_length),
                                                             initializer="uniform",
                                                             name="node_1_input_kernel")
        self.candidate_dnn_hidden_bias_2 = self.add_weight(shape=(c.STATE_EXTENSION**3 * self.state_length,),
                                                           initializer="uniform",
                                                           name="node_1_input_bias")

        self.candidate_dnn_hidden_kernel_3 = self.add_weight(shape=(c.STATE_EXTENSION**3 * self.state_length,
                                                                    c.STATE_EXTENSION**2 * self.state_length),
                                                             initializer="uniform",
                                                             name="node_1_input_kernel")
        self.candidate_dnn_hidden_bias_3 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,),
                                                           initializer="uniform",
                                                           name="node_1_input_bias")

        self.candidate_dnn_hidden_kernel_4 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,
                                                                    c.STATE_EXTENSION * self.state_length),
                                                             initializer="uniform",
                                                             name="node_1_input_kernel")
        self.candidate_dnn_hidden_bias_4 = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,),
                                                           initializer="uniform",
                                                           name="node_1_input_bias")

        self.candidate_dnn_output_kernel = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,
                                                                  self.state_length),
                                                           initializer="uniform",
                                                           name="node_1_output_kernel")
        self.candidate_dnn_output_bias = self.add_weight(shape=(self.state_length,),
                                                         initializer="uniform",
                                                         name="node_1_output_bias")

    def _call_candidate_dnn(self, inputs):
        outputs = tf.matmul(inputs, self.candidate_dnn_input_kernel) + self.candidate_dnn_input_bias
        outputs = tf.matmul(outputs, self.candidate_dnn_hidden_kernel_1) + self.candidate_dnn_hidden_bias_1
        outputs = tf.tanh(tf.matmul(outputs, self.candidate_dnn_hidden_kernel_2) + self.candidate_dnn_hidden_bias_2)
        outputs = tf.tanh(tf.matmul(outputs, self.candidate_dnn_hidden_kernel_3) + self.candidate_dnn_hidden_bias_3)
        outputs = tf.matmul(outputs, self.candidate_dnn_hidden_kernel_4) + self.candidate_dnn_hidden_bias_4
        outputs = tf.matmul(outputs, self.candidate_dnn_output_kernel) + self.candidate_dnn_output_bias
        return outputs

    def _build_forget_dnn(self, input_shape):
        self.forget_dnn_input_kernel = self.add_weight(shape=(input_shape + self.state_length,
                                                              c.STATE_EXTENSION * self.state_length),
                                                       initializer="uniform",
                                                       name="node_2_input_kernel")
        self.forget_dnn_input_bias = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,),
                                                     initializer="uniform",
                                                     name="node_2_input_bias")

        self.forget_dnn_hidden_kernel_1 = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,
                                                                 c.STATE_EXTENSION**2 * self.state_length),
                                                          initializer="uniform",
                                                          name="node_2_hidden_kernel_1")
        self.forget_dnn_hidden_bias_1 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,),
                                                        initializer="uniform",
                                                        name="node_2_hidden_bias_1")

        self.forget_dnn_hidden_kernel_2 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,
                                                                 c.STATE_EXTENSION**3 * self.state_length),
                                                          initializer="uniform",
                                                          name="node_2_hidden_kernel_2")
        self.forget_dnn_hidden_bias_2 = self.add_weight(shape=(c.STATE_EXTENSION**3 * self.state_length,),
                                                        initializer="uniform",
                                                        name="node_2_hidden_bias_2")

        self.forget_dnn_hidden_kernel_3 = self.add_weight(shape=(c.STATE_EXTENSION**3 * self.state_length,
                                                                 c.STATE_EXTENSION**2 * self.state_length),
                                                          initializer="uniform",
                                                          name="node_2_hidden_kernel_2")
        self.forget_dnn_hidden_bias_3 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,),
                                                        initializer="uniform",
                                                        name="node_2_hidden_bias_2")

        self.forget_dnn_hidden_kernel_4 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,
                                                                 c.STATE_EXTENSION * self.state_length),
                                                          initializer="uniform",
                                                          name="node_2_hidden_kernel_2")
        self.forget_dnn_hidden_bias_4 = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,),
                                                        initializer="uniform",
                                                        name="node_2_hidden_bias_2")

        self.forget_dnn_output_kernel = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,
                                                               self.state_length),
                                                        initializer="uniform",
                                                        name="node_2_output_kernel")
        self.forget_dnn_output_bias = self.add_weight(shape=(self.state_length,),
                                                      initializer="uniform",
                                                      name="node_2_output_bias")

    def _call_forget_dnn(self, inputs):
        outputs = tf.matmul(inputs, self.forget_dnn_input_kernel) + self.forget_dnn_input_bias
        outputs = tf.tanh(tf.matmul(outputs, self.forget_dnn_hidden_kernel_1) + self.forget_dnn_hidden_bias_1)
        outputs = tf.tanh(tf.matmul(outputs, self.forget_dnn_hidden_kernel_2) + self.forget_dnn_hidden_bias_2)
        outputs = tf.sigmoid(tf.matmul(outputs, self.forget_dnn_hidden_kernel_3) + self.forget_dnn_hidden_bias_3)
        outputs = tf.sigmoid(tf.matmul(outputs, self.forget_dnn_hidden_kernel_4) + self.forget_dnn_hidden_bias_4)
        outputs = tf.sigmoid(tf.matmul(outputs, self.forget_dnn_output_kernel) + self.forget_dnn_output_bias)
        return outputs

    def _build_output_dnn(self):
        self.output_dnn_input_kernel = self.add_weight(shape=(self.state_length,
                                                              c.STATE_EXTENSION * self.state_length),
                                                       initializer="uniform",
                                                       name="node_3_input_kernel")
        self.output_dnn_input_bias = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,),
                                                     initializer="uniform",
                                                     name="node_3_input_bias")

        self.output_dnn_hidden_kernel_1 = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,
                                                                 c.STATE_EXTENSION**2 * self.state_length),
                                                          initializer="uniform",
                                                          name="node_3_hidden_kernel_1")
        self.output_dnn_hidden_bias_1 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,),
                                                        initializer="uniform",
                                                        name="node_3_hidden_bias_1")

        self.output_dnn_hidden_kernel_2 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,
                                                                 c.STATE_EXTENSION**3 * self.state_length),
                                                          initializer="uniform",
                                                          name="node_3_hidden_kernel_2")
        self.output_dnn_hidden_bias_2 = self.add_weight(shape=(c.STATE_EXTENSION**3 * self.state_length,),
                                                        initializer="uniform",
                                                        name="node_3_hidden_bias_2")

        self.output_dnn_hidden_kernel_3 = self.add_weight(shape=(c.STATE_EXTENSION**3 * self.state_length,
                                                                 c.STATE_EXTENSION**2 * self.state_length),
                                                          initializer="uniform",
                                                          name="node_3_hidden_kernel_1")
        self.output_dnn_hidden_bias_3 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,),
                                                        initializer="uniform",
                                                        name="node_3_hidden_bias_1")

        self.output_dnn_hidden_kernel_4 = self.add_weight(shape=(c.STATE_EXTENSION**2 * self.state_length,
                                                                 c.STATE_EXTENSION * self.state_length),
                                                          initializer="uniform",
                                                          name="node_3_hidden_kernel_2")
        self.output_dnn_hidden_bias_4 = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,),
                                                        initializer="uniform",
                                                        name="node_3_hidden_bias_2")

        self.output_dnn_output_kernel = self.add_weight(shape=(c.STATE_EXTENSION * self.state_length,
                                                               self.number_of_output),
                                                        initializer="uniform",
                                                        name="node_3_output_kernel")
        self.output_dnn_output_bias = self.add_weight(shape=(self.number_of_output,),
                                                      initializer="uniform",
                                                      name="node_3_output_bias")

    def _call_output_dnn(self, inputs):
        outputs = tf.matmul(inputs, self.output_dnn_input_kernel) + self.output_dnn_input_bias
        outputs = tf.matmul(outputs, self.output_dnn_hidden_kernel_1) + self.output_dnn_hidden_bias_1
        outputs = tf.tanh(tf.matmul(outputs, self.output_dnn_hidden_kernel_2) + self.output_dnn_hidden_bias_2)
        outputs = tf.tanh(tf.matmul(outputs, self.output_dnn_hidden_kernel_3) + self.output_dnn_hidden_bias_3)
        outputs = tf.matmul(outputs, self.output_dnn_hidden_kernel_4) + self.output_dnn_hidden_bias_4
        outputs = tf.matmul(outputs, self.output_dnn_output_kernel) + self.output_dnn_output_bias
        return outputs
