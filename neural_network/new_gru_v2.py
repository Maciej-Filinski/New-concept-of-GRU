from tensorflow import keras
import tensorflow as tf


class NewGRU2(keras.layers.Layer):
    def __init__(self,
                 state_length: int,
                 number_of_outputs: int,
                 candidate_dnn_structure: dict,
                 forget_dnn_structure: dict,
                 output_dnn_structure: dict,
                 forget_dnn_enable=True,
                 output_dnn_enable=True,
                 return_full_output=False,
                 **kwargs):
        self.state_length = state_length
        self.state_size = [tf.TensorShape(state_length)]
        self.number_of_outputs = number_of_outputs
        self.candidate_dnn_structure = candidate_dnn_structure
        self.candidate_dnn = []
        self.forget_dnn_structure = forget_dnn_structure
        self.forget_dnn = []
        self.output_dnn_structure = output_dnn_structure
        self.output_dnn = []
        self.forget_dnn_enable = forget_dnn_enable
        self.output_dnn_enable = output_dnn_enable
        self.return_full_output = return_full_output
        super(NewGRU2, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1]
        self._build_candidate_dnn(input_shape)
        if self.forget_dnn_enable:
            self._build_forget_dnn(input_shape)
        if self.output_dnn_enable:
            self._build_output_dnn()

    def call(self, inputs, states):
        states = states[0]
        inputs = tf.concat([inputs, states], -1)
        candidate_dnn_outputs = self._call_candidate_dnn(inputs)
        state_correction = tf.subtract(candidate_dnn_outputs, states)
        if self.forget_dnn_enable:
            forget_dnn_outputs = self._call_forget_dnn(inputs)
            state_correction = tf.math.multiply(0.8, state_correction)
        else:
            forget_dnn_outputs = None
        new_states = tf.add(state_correction, states)
        if self.output_dnn_enable:
            outputs = self._call_output_dnn(new_states)
        else:
            outputs = new_states
        if self.return_full_output:
            return (outputs, new_states, forget_dnn_outputs, candidate_dnn_outputs), new_states
        else:
            return outputs, new_states

    def _build_candidate_dnn(self, input_shape):
        input_shape = input_shape + self.state_length
        for key in self.candidate_dnn_structure:
            kernel = self.add_weight(shape=(input_shape, self.candidate_dnn_structure[key]),
                                     initializer='uniform',
                                     name=key + '_kernel')
            bias = self.add_weight(shape=(self.candidate_dnn_structure[key],),
                                   initializer='uniform',
                                   name=key + '_bias')
            self.candidate_dnn.append([kernel, bias])
            input_shape = self.candidate_dnn_structure[key]

    def _call_candidate_dnn(self, inputs):
        outputs = inputs
        layer_number = 1
        for kernel, bias in self.candidate_dnn:
            if layer_number == len(self.candidate_dnn) or layer_number == 1:
                outputs = tf.matmul(outputs, kernel) + bias
            else:
                outputs = tf.tanh(tf.matmul(outputs, kernel)) + bias
            layer_number += 1
        return outputs

    def _build_forget_dnn(self, input_shape):
        input_shape = input_shape + self.state_length
        for key in self.forget_dnn_structure:
            kernel = self.add_weight(shape=(input_shape, self.forget_dnn_structure[key]),
                                     initializer='uniform',
                                     name=key + '_kernel')
            bias = self.add_weight(shape=(self.forget_dnn_structure[key],),
                                   initializer='uniform',
                                   name=key + '_bias')
            self.forget_dnn.append([kernel, bias])
            input_shape = self.forget_dnn_structure[key]

    def _call_forget_dnn(self, inputs):
        outputs = inputs
        layer_number = 1
        for kernel, bias in self.forget_dnn:
            if layer_number == len(self.forget_dnn):
                outputs = tf.sigmoid(tf.matmul(outputs, kernel)) + bias
            else:
                outputs = tf.tanh(tf.matmul(outputs, kernel)) + bias
            layer_number += 1
        return outputs

    def _build_output_dnn(self):
        input_shape = self.state_length
        layer_number = 1
        for key in self.output_dnn_structure:
            if layer_number == len(self.output_dnn_structure):
                kernel = self.add_weight(shape=(input_shape, self.number_of_outputs),
                                         initializer='uniform',
                                         name=key + '_kernel')
                bias = self.add_weight(shape=(self.number_of_outputs,),
                                       initializer='uniform',
                                       name=key + '_bias')
            else:
                kernel = self.add_weight(shape=(input_shape, self.output_dnn_structure[key]),
                                         initializer='uniform',
                                         name=key + '_kernel')
                bias = self.add_weight(shape=(self.output_dnn_structure[key],),
                                       initializer='uniform',
                                       name=key + '_bias')
            self.output_dnn.append([kernel, bias])
            input_shape = self.output_dnn_structure[key]
            layer_number += 1

    def _call_output_dnn(self, inputs):
        outputs = inputs
        layer_number = 1
        for kernel, bias in self.output_dnn:
            if layer_number == len(self.output_dnn_structure) or layer_number == 1:
                outputs = tf.matmul(outputs, kernel) + bias
            else:
                outputs = tf.tanh(tf.matmul(outputs, kernel)) + bias
            layer_number += 1
        return outputs
