from NewGRU import NewGRU
import tensorflow as tf
from SystemDefinition import LinearSystem
import matplotlib.pyplot as plt
import numpy as np

def create_model(number_of_system_input, state_length, number_of_system_output):
    model = NewGRU(input_shape=(1, number_of_system_input),
                   state_length=state_length,
                   output_length=number_of_system_output)
    return model


def train_model(inputs, outputs, neural_network_model: NewGRU):
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    neural_network_model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(inputs, outputs, epochs=10)


if __name__ == "__main__":
    number_of_system_input = 10
    state_length = 3
    number_of_system_output = 10
    model = create_model(number_of_system_input, state_length, number_of_system_output)
    x = tf.random.uniform(shape=(1, number_of_system_input))
    print(model(x))
    print(model(x))

    train_inputs = tf.random.uniform(shape=(1000, 1, number_of_system_input))
    train_outputs = tf.multiply(train_inputs, train_inputs)
    train_model(train_inputs, train_outputs, model)
    test_input = tf.random.uniform(shape=(10000, 1, number_of_system_input))
    test_output = tf.multiply(test_input, test_input)
    predict_output = np.zeros(test_input.shape)
    print(test_input[1, :, :])
    model(test_input[1, :, :])
    for i in range(10000):
        predict_output[i, :, :] = model(test_input[i, :, :])
        #print(predict_output[i, :, :])
    print(f' test input = {test_input.shape}')
    print(f' predict = {predict_output.shape}')
    print(f' test input = {test_output.shape}')

    plt.plot(test_output[:, :, 0], label='real system')
    plt.plot(predict_output[:, :, 0], label='predict output')
    plt.legend()
    plt.show()

