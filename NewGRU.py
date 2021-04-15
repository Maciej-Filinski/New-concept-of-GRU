from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


class NewGRU(keras.layers.Layer):
    def __init__(self, unit, **kwargs):
        self.unit = unit
        self.state_size = [tf.TensorShape(unit)]
        super(NewGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1]
        self.kernel = self.add_weight(shape=(input_shape + self.unit, self.unit), initializer="uniform",
                                      name="kernel")
        self.recurrent_kernel = self.add_weight(shape=(self.unit, self.unit), initializer="uniform",
                                                name="recurrent_kernel")

    def call(self, inputs, states):
        states = states[0]
        inputs = tf.concat([inputs, states], -1)
        output_1 = tf.matmul(inputs, self.kernel)
        output_2 = tf.matmul(states, self.recurrent_kernel)
        new_states = states + output_1 + output_2
        tf.print('output = ', inputs, 'old NN state = ', states, 'new NN state = ', new_states)
        return output_1, new_states
