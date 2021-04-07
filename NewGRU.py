import tensorflow as tf
from tensorflow.keras import layers
from abc import ABC


class NewGRU(tf.keras.Model, ABC):
    def __init__(self, input_shape, state_length, *args, **kwargs):
        super(NewGRU, self).__init__(*args, **kwargs)
        ''' Create node 1 (DNN) '''
        self.node_1 = layers.Dense(state_length, input_shape=input_shape, activation='sigmoid')

        ''' Create node 2 (DNN) '''
        self.node_2 = layers.Dense(state_length, input_shape=input_shape, activation='sigmoid')

        ''' Create node 3 (DNN) '''
        self.node_3 = layers.Dense(state_length, activation='sigmoid')

        self.recurrent_state = tf.Variable(initial_value=tf.zeros(shape=(1, state_length)),
                                           trainable=False,
                                           validate_shape=False,
                                           dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        network_input = tf.keras.layers.Concatenate(axis=-1)([inputs, self.recurrent_state])

        node_1 = self.node_1(network_input)

        node_2 = self.node_2(network_input)

        tf.print(self.recurrent_state)
        tf.print(self.recurrent_state.shape)

        hat_ht = tf.math.subtract(node_1, self.recurrent_state)
        f_t = tf.math.multiply(node_2, hat_ht)
        self.recurrent_state.assign(tf.math.add(f_t, self.recurrent_state))

        tf.print(self.recurrent_state)
        tf.print(self.recurrent_state.shape)

        node_3 = self.node_3(self.recurrent_state)
        return node_3
