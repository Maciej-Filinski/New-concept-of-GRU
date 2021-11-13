import tensorflow as tf
from tensorflow import keras


class BasicLSTM(keras.layers.Layer):
    def __init__(self, state_length, **kwargs):
        self.state_length = state_length
        self.state_size = [tf.TensorShape(state_length), tf.TensorShape(state_length)]
        super(BasicLSTM, self).__init__(**kwargs)
        self.w_f = None
        self.u_f = None
        self.b_f = None
        self.w_i = None
        self.u_i = None
        self.b_i = None
        self.w_o = None
        self.u_o = None
        self.b_o = None
        self.w_c = None
        self.u_c = None
        self.b_c = None

    def build(self, input_shape):
        input_shape = input_shape[1]
        # self.w_f = self.add_weight(shape=(input_shape, self.state_length), initializer='uniform', name='wf')
        # self.u_f = self.add_weight(shape=(self.state_length, self.state_length), initializer='uniform', name='uf')
        # self.b_f = self.add_weight(shape=(self.state_length,), initializer='uniform', name='bf')
        # self.w_i = self.add_weight(shape=(input_shape, self.state_length), initializer='uniform', name='wi')
        # self.u_i = self.add_weight(shape=(self.state_length, self.state_length), initializer='uniform', name='ui')
        # self.b_i = self.add_weight(shape=(self.state_length,), initializer='uniform', name='bi')
        # self.w_o = self.add_weight(shape=(input_shape, self.state_length), initializer='uniform', name='wo')
        # self.u_o = self.add_weight(shape=(self.state_length, self.state_length), initializer='uniform', name='uo')
        # self.b_o = self.add_weight(shape=(self.state_length,), initializer='uniform', name='bo')
        # self.w_c = self.add_weight(shape=(input_shape, self.state_length), initializer='uniform', name='wc')
        # self.u_c = self.add_weight(shape=(self.state_length, self.state_length), initializer='uniform', name='uc')
        # self.b_c = self.add_weight(shape=(self.state_length,), initializer='uniform', name='bc')
        self.kernel = self.add_weight(shape=(input_shape, 4 * self.state_length), initializer='uniform', name='w')
        self.recurrent_kernel = self.add_weight(shape=(self.state_length, 4 * self.state_length), initializer='uniform', name='u')
        self.bias = self.add_weight(shape=(4 * self.state_length,), initializer='uniform', name='b')


    def call(self, inputs, states):
        # cell_states = states[1]
        # outputs_states = states[0]
        # forget_gate = tf.sigmoid(tf.matmul(inputs, self.w_f) + tf.matmul(outputs_states, self.u_f) + self.b_f)
        # input_gate = tf.sigmoid(tf.matmul(inputs, self.w_i) + tf.matmul(outputs_states, self.u_i) + self.b_i)
        # outputs_gate = tf.sigmoid(tf.matmul(inputs, self.w_o) + tf.matmul(outputs_states, self.u_o) + self.b_o)
        # input_cell = tf.sigmoid(tf.matmul(inputs, self.w_c) + tf.matmul(outputs_states, self.u_c) + self.b_c)
        # cell_states = tf.math.multiply(forget_gate, cell_states) + tf.math.multiply(input_cell, input_gate)
        # outputs_states = tf.math.multiply(outputs_gate, tf.tanh(cell_states))
        # return outputs_states, [outputs_states, cell_states]
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        z = tf.keras.backend.dot(inputs, self.kernel)
        z += tf.keras.backend.dot(h_tm1, self.recurrent_kernel)
        z = tf.keras.backend.bias_add(z, self.bias)

        z0, z1, z2, z3 = tf.split(z, 4, axis=1)

        i = tf.sigmoid(z0)
        f = tf.sigmoid(z1)
        c = f * c_tm1 + i * tf.tanh(z2)
        o = tf.sigmoid(z3)

        h = o * tf.tanh(c)
        return h, [h, c]


if __name__ == '__main__':
    cell = BasicLSTM(state_length=8)
    rnn = keras.layers.RNN(cell, stateful=True)
    input_layer = keras.Input(batch_shape=(1, None, 1), ragged=True)
    output = rnn(input_layer)
    model = keras.models.Model(input_layer, output)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
