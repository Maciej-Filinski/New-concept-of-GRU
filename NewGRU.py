import tensorflow as tf
from tensorflow.keras import layers
from abc import ABC


class NewGRU(tf.keras.Model, ABC):
    def __init__(self, input_shape, state_length, output_length, *args, **kwargs):
        super(NewGRU, self).__init__(*args, **kwargs)
        ''' Create node 1 (DNN) '''
        self.node_1_input_layer = layers.Dense(32, input_shape=input_shape, activation='sigmoid')
        self.node_1_output_layer = layers.Dense(state_length, activation='sigmoid')

        ''' Create node 2 (DNN) '''
        self.node_2_input_layer = layers.Dense(32, input_shape=input_shape, activation='sigmoid')
        self.node_2_output_layer = layers.Dense(state_length, activation='sigmoid')

        ''' Create node 3 (DNN) '''
        self.node_3_input_layer = layers.Dense(32, activation='sigmoid')
        self.node_3_output_layer = layers.Dense(output_length, activation='sigmoid')

        self.recurrent_state = tf.Variable(initial_value=tf.zeros(shape=(1, state_length)),
                                           trainable=False,
                                           validate_shape=False,
                                           dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        if inputs.shape[0] is None:
            inputs = inputs[0]
        network_input = tf.keras.layers.Concatenate(axis=-1)([inputs, self.recurrent_state])

        node_1 = self.node_1_input_layer(network_input)

        node_1 = self.node_1_output_layer(node_1)

        node_2 = self.node_2_input_layer(network_input)
        node_2 = self.node_2_output_layer(node_2)

        #tf.print(self.recurrent_state)
        #tf.print(self.recurrent_state.shape)

        hat_ht = tf.math.subtract(node_1, self.recurrent_state)
        f_t = tf.math.multiply(node_2, hat_ht)
        self.recurrent_state.assign(tf.math.add(f_t, self.recurrent_state))

        #tf.print(self.recurrent_state)
        #tf.print(self.recurrent_state.shape)

        node_3 = self.node_3_input_layer(self.recurrent_state)
        node_3 = self.node_3_output_layer(node_3)
        return node_3
