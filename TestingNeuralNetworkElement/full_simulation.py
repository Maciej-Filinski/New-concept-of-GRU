import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_generator import *
from build_model import *

EPOCHS = 10
TRAIN_DATA_LENGTH = 1000
TEST_DATA_LENGTH = 100
BATCH_SIZE = 20
open("./Result/result_data.txt", 'w').close()

if __name__ == "__main__":
    ''' Data generation '''
    data = data_linear_system_input_state(number_of_sample_train=TRAIN_DATA_LENGTH,
                                          number_of_sample_test=TEST_DATA_LENGTH)
    x_train_linear_is, y_train_linear_is, x_test_linear_is, y_test_linear_is = data
    data = data_linear_system_input_output(number_of_sample_train=TRAIN_DATA_LENGTH,
                                           number_of_sample_test=TEST_DATA_LENGTH)
    x_train_linear_io, y_train_linear_io, x_test_linear_io, y_test_linear_io = data
    data = data_nonlinear_system_input_state(number_of_sample_train=TRAIN_DATA_LENGTH,
                                             number_of_sample_test=TEST_DATA_LENGTH)
    x_train_nonlinear_is, y_train_nonlinear_is, x_test_nonlinear_is, y_test_nonlinear_is = data
    data = data_nonlinear_system_input_output(number_of_sample_train=TRAIN_DATA_LENGTH,
                                              number_of_sample_test=TEST_DATA_LENGTH)
    x_train_nonlinear_io, y_train_nonlinear_io, x_test_nonlinear_io, y_test_nonlinear_io = data

    for i in range(3):
        ''' Linear system noiseless. Input -> State Space Vector '''
        model = build_model_input_state(ss_vector_length=2, number_of_inputs=1, batch_size=BATCH_SIZE, struct=i+1)
        model.summary()
        x_train = tf.ragged.constant([x_train_linear_is[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
        y_train = y_train_linear_is[0:TRAIN_DATA_LENGTH, :]
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        weights = model.get_weights()
        model.save_weights('./Result/models/linear_input_state/structure_' + str(i + 1) + '/weights')
        with open("./Result/result_data.txt", "a") as fp:
            fp.write('linear_input_state_structure_' + str(i + 1))
            fp.write(str(history.history))
            fp.write('\n')

        model = build_model_input_state(ss_vector_length=2, number_of_inputs=1, batch_size=1, struct=i+1)
        model.set_weights(weights)
        model.reset_states()
        predicted_train_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 2))
        for j in range(TEST_DATA_LENGTH):
            predicted_train_outputs[j, :] = model.predict(x_train_linear_is[j, :, :])

        model.reset_states()
        predicted_test_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 2))
        for j in range(TEST_DATA_LENGTH):
            predicted_test_outputs[j, :] = model.predict(x_test_linear_is[j, :, :])
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(y_train_linear_is[:TEST_DATA_LENGTH, 0], 'g', label='true state', linewidth=2)
        axs[0, 0].plot(predicted_train_outputs[:, 0], '--r', label='predicted state', linewidth=2)
        axs[0, 0].set_title('train: x_1')
        axs[0, 0].legend()
        axs[0, 1].plot(y_train_linear_is[:TEST_DATA_LENGTH, 1], '--g', label='true state', linewidth=2)
        axs[0, 1].plot(predicted_train_outputs[:, 1], '--r', label='predicted state', linewidth=2)
        axs[0, 1].set_title('train: x_2')
        axs[0, 1].legend()
        axs[1, 0].plot(y_test_linear_is[:, 0], 'b', label='true state', linewidth=4)
        axs[1, 0].plot(predicted_test_outputs[:, 0], '--r', label='predicted state', linewidth=2)
        axs[1, 0].set_title('test: x_1 - impulse response')
        axs[1, 0].legend()
        axs[1, 1].plot(y_test_linear_is[:, 1], 'b', label='true state', linewidth=4)
        axs[1, 1].plot(predicted_test_outputs[:, 1], '--r', label='predicted state', linewidth=2)
        axs[1, 1].set_title('test: x_2 - impulse response')
        axs[1, 1].legend()
        plt.savefig('./Result/figures/linear_input_state_structure_' + str(i + 1))

        ''' Linear system noise. Input -> State Space Vector '''
        model = build_model_input_state(ss_vector_length=2, number_of_inputs=1, batch_size=BATCH_SIZE, struct=i+1)
        model.summary()
        x_train = tf.ragged.constant([x_train_linear_is[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
        y_train = np.copy(y_train_linear_is[0:TRAIN_DATA_LENGTH, :])
        y_train += np.random.normal(0, 0.5, size=(TRAIN_DATA_LENGTH, 2))
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        weights = model.get_weights()
        model.save_weights('./Result/models/linear_input_state_noise/structure_' + str(i + 1) + '/weights')
        with open("./Result/result_data.txt", "a") as fp:
            fp.write('linear_input_state_noise_structure_' + str(i + 1))
            fp.write(str(history.history))
            fp.write('\n')

        model = build_model_input_state(ss_vector_length=2, number_of_inputs=1, batch_size=1, struct=i+1)
        model.set_weights(weights)
        model.reset_states()
        predicted_train_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 2))
        for j in range(TEST_DATA_LENGTH):
            predicted_train_outputs[j, :] = model.predict(x_train_linear_is[j, :, :])

        model.reset_states()
        predicted_test_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 2))
        for j in range(TEST_DATA_LENGTH):
            predicted_test_outputs[j, :] = model.predict(x_test_linear_is[j, :, :])
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(y_train_linear_is[:TEST_DATA_LENGTH, 0], 'g', label='true state', linewidth=2)
        axs[0, 0].plot(predicted_train_outputs[:, 0], '--r', label='predicted state', linewidth=2)
        axs[0, 0].set_title('train: x_1')
        axs[0, 0].legend()
        axs[0, 1].plot(y_train_linear_is[:TEST_DATA_LENGTH, 1], '--g', label='true state', linewidth=2)
        axs[0, 1].plot(predicted_train_outputs[:, 1], '--r', label='predicted state', linewidth=2)
        axs[0, 1].set_title('train: x_2')
        axs[0, 1].legend()
        axs[1, 0].plot(y_test_linear_is[:, 0], 'b', label='true state', linewidth=4)
        axs[1, 0].plot(predicted_test_outputs[:, 0], '--r', label='predicted state', linewidth=2)
        axs[1, 0].set_title('test: x_1 - impulse response')
        axs[1, 0].legend()
        axs[1, 1].plot(y_test_linear_is[:, 1], 'b', label='true state', linewidth=4)
        axs[1, 1].plot(predicted_test_outputs[:, 1], '--r', label='predicted state', linewidth=2)
        axs[1, 1].set_title('test: x_2 - impulse response')
        axs[1, 1].legend()
        plt.savefig('./Result/figures/linear_input_state_noise_structure_' + str(i + 1))

        ''' Nonlinear system noiseless. Input -> State Space Vector '''
        model = build_model_input_state(ss_vector_length=2, number_of_inputs=1, batch_size=BATCH_SIZE, struct=i+1)
        model.summary()
        x_train = tf.ragged.constant([x_train_nonlinear_is[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
        y_train = y_train_nonlinear_is[0:TRAIN_DATA_LENGTH, :]
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        weights = model.get_weights()
        model.save_weights('./Result/models/nonlinear_input_state/structure_' + str(i + 1) + '/weights')
        with open("./Result/result_data.txt", "a") as fp:
            fp.write('nonlinear_input_state_structure_' + str(i + 1))
            fp.write(str(history.history))
            fp.write('\n')

        model = build_model_input_state(ss_vector_length=2, number_of_inputs=1, batch_size=1, struct=i+1)
        model.set_weights(weights)
        model.reset_states()
        predicted_train_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 2))
        for j in range(TEST_DATA_LENGTH):
            predicted_train_outputs[j, :] = model.predict(x_train_nonlinear_is[j, :, :])

        model.reset_states()
        predicted_test_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 2))
        for j in range(TEST_DATA_LENGTH):
            predicted_test_outputs[j, :] = model.predict(x_test_nonlinear_is[j, :, :])
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(y_train_nonlinear_is[:TEST_DATA_LENGTH, 0], 'g', label='true state', linewidth=2)
        axs[0, 0].plot(predicted_train_outputs[:, 0], '--r', label='predicted state', linewidth=2)
        axs[0, 0].set_title('train: x_1')
        axs[0, 0].legend()
        axs[0, 1].plot(y_train_nonlinear_is[:TEST_DATA_LENGTH, 1], '--g', label='true state', linewidth=2)
        axs[0, 1].plot(predicted_train_outputs[:, 1], '--r', label='predicted state', linewidth=2)
        axs[0, 1].set_title('train: x_2')
        axs[0, 1].legend()
        axs[1, 0].plot(y_test_nonlinear_is[:, 0], 'b', label='true state', linewidth=4)
        axs[1, 0].plot(predicted_test_outputs[:, 0], '--r', label='predicted state', linewidth=2)
        axs[1, 0].set_title('test: x_1 - impulse response')
        axs[1, 0].legend()
        axs[1, 1].plot(y_test_nonlinear_is[:, 1], 'b', label='true state', linewidth=4)
        axs[1, 1].plot(predicted_test_outputs[:, 1], '--r', label='predicted state', linewidth=2)
        axs[1, 1].set_title('test: x_2 - impulse response')
        axs[1, 1].legend()
        plt.savefig('./Result/figures/nonlinear_input_state_structure_' + str(i + 1))

        ''' Nonlinear system noise. Input -> State Space Vector '''
        model = build_model_input_state(ss_vector_length=2, number_of_inputs=1, batch_size=BATCH_SIZE, struct=i+1)
        model.summary()
        x_train = tf.ragged.constant([x_train_nonlinear_is[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
        y_train = np.copy(y_train_nonlinear_is[0:TRAIN_DATA_LENGTH, :])
        y_train += np.random.normal(0, 0.5, size=(TRAIN_DATA_LENGTH, 2))
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        weights = model.get_weights()
        model.save_weights('./Result/models/nonlinear_input_state_noise/structure_' + str(i + 1) + '/weights')
        with open("./Result/result_data.txt", "a") as fp:
            fp.write('nonlinear_input_state_noise_structure_' + str(i + 1))
            fp.write(str(history.history))
            fp.write('\n')

        model = build_model_input_state(ss_vector_length=2, number_of_inputs=1, batch_size=1, struct=i+1)
        model.set_weights(weights)
        model.reset_states()
        predicted_train_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 2))
        for j in range(TEST_DATA_LENGTH):
            predicted_train_outputs[j, :] = model.predict(x_train_nonlinear_is[j, :, :])

        model.reset_states()
        predicted_test_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 2))
        for j in range(TEST_DATA_LENGTH):
            predicted_test_outputs[j, :] = model.predict(x_test_nonlinear_is[j, :, :])
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(y_train_nonlinear_is[:TEST_DATA_LENGTH, 0], 'g', label='true state', linewidth=2)
        axs[0, 0].plot(predicted_train_outputs[:, 0], '--r', label='predicted state', linewidth=2)
        axs[0, 0].set_title('train: x_1')
        axs[0, 0].legend()
        axs[0, 1].plot(y_train_nonlinear_is[:TEST_DATA_LENGTH, 1], '--g', label='true state', linewidth=2)
        axs[0, 1].plot(predicted_train_outputs[:, 1], '--r', label='predicted state', linewidth=2)
        axs[0, 1].set_title('train: x_2')
        axs[0, 1].legend()
        axs[1, 0].plot(y_test_nonlinear_is[:, 0], 'b', label='true state', linewidth=4)
        axs[1, 0].plot(predicted_test_outputs[:, 0], '--r', label='predicted state', linewidth=2)
        axs[1, 0].set_title('test: x_1 - impulse response')
        axs[1, 0].legend()
        axs[1, 1].plot(y_test_nonlinear_is[:, 1], 'b', label='true state', linewidth=4)
        axs[1, 1].plot(predicted_test_outputs[:, 1], '--r', label='predicted state', linewidth=2)
        axs[1, 1].set_title('test: x_2 - impulse response')
        axs[1, 1].legend()
        plt.savefig('./Result/figures/nonlinear_input_state_noise_structure_' + str(i + 1))










        ''' Linear system noiseless. Input -> State Space Vector '''
        model = build_model_input_output(number_of_outputs=1, number_of_inputs=1, batch_size=BATCH_SIZE, struct=i+1)
        model.summary()
        x_train = tf.ragged.constant([x_train_linear_io[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
        y_train = y_train_linear_io[0:TRAIN_DATA_LENGTH, :]
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        weights = model.get_weights()
        model.save_weights('./Result/models/linear_input_output/structure_' + str(i + 1) + '/weights')
        with open("./Result/result_data.txt", "a") as fp:
            fp.write('linear_input_output_structure_' + str(i + 1))
            fp.write(str(history.history))
            fp.write('\n')

        model = build_model_input_output(number_of_outputs=1, number_of_inputs=1, batch_size=1, struct=i+1)
        model.set_weights(weights)
        model.reset_states()
        predicted_train_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 1))
        for j in range(TEST_DATA_LENGTH):
            predicted_train_outputs[j, :] = model.predict(x_train_linear_io[j, :, :])

        model.reset_states()
        predicted_test_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 1))
        for j in range(TEST_DATA_LENGTH):
            predicted_test_outputs[j, :] = model.predict(x_test_linear_io[j, :, :])
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(y_train_linear_io[:TEST_DATA_LENGTH, 0], 'g', label='true output', linewidth=2)
        axs[0].plot(predicted_train_outputs[:, 0], '--r', label='predicted output', linewidth=2)
        axs[0].set_title('train: x_1')
        axs[0].legend()
        axs[1].plot(y_test_linear_io[:TEST_DATA_LENGTH, 0], '--g', label='true output', linewidth=2)
        axs[1].plot(predicted_test_outputs[:, 0], '--r', label='predicted output', linewidth=2)
        axs[1].set_title('train: x_2')
        axs[1].legend()
        plt.savefig('./Result/figures/linear_input_output_structure_' + str(i + 1))

        ''' Linear system noise. Input -> State Space Vector '''
        model = build_model_input_output(number_of_outputs=1, number_of_inputs=1, batch_size=BATCH_SIZE, struct=i+1)
        model.summary()
        x_train = tf.ragged.constant([x_train_linear_io[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
        y_train = np.copy(y_train_linear_io[0:TRAIN_DATA_LENGTH, :])
        y_train += np.random.normal(0, 0.5, size=(TRAIN_DATA_LENGTH, 1))
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        weights = model.get_weights()
        model.save_weights('./Result/models/linear_input_output_noise/structure_' + str(i + 1) + '/weights')
        with open("./Result/result_data.txt", "a") as fp:
            fp.write('linear_input_output_noise_structure_' + str(i + 1))
            fp.write(str(history.history))
            fp.write('\n')

        model = build_model_input_output(number_of_outputs=1, number_of_inputs=1, batch_size=1, struct=i+1)
        model.set_weights(weights)
        model.reset_states()
        predicted_train_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 1))
        for j in range(TEST_DATA_LENGTH):
            predicted_train_outputs[j, :] = model.predict(x_train_linear_io[j, :, :])

        model.reset_states()
        predicted_test_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 1))
        for j in range(TEST_DATA_LENGTH):
            predicted_test_outputs[j, :] = model.predict(x_test_linear_io[j, :, :])
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(y_train_linear_io[:TEST_DATA_LENGTH, 0], 'g', label='true output', linewidth=2)
        axs[0].plot(predicted_train_outputs[:, 0], '--r', label='predicted output', linewidth=2)
        axs[0].set_title('train: x_1')
        axs[0].legend()
        axs[1].plot(y_test_linear_io[:TEST_DATA_LENGTH, 0], '--g', label='true output', linewidth=2)
        axs[1].plot(predicted_test_outputs[:, 0], '--r', label='predicted output', linewidth=2)
        axs[1].set_title('train: x_2')
        axs[1].legend()
        plt.savefig('./Result/figures/linear_input_output_noise_structure_' + str(i + 1))

        ''' Nonlinear system noiseless. Input -> State Space Vector '''
        model = build_model_input_output(number_of_outputs=1, number_of_inputs=1, batch_size=BATCH_SIZE, struct=i+1)
        model.summary()
        x_train = tf.ragged.constant([x_train_nonlinear_io[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
        y_train = y_train_nonlinear_io[0:TRAIN_DATA_LENGTH, :]
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        weights = model.get_weights()
        model.save_weights('./Result/models/nonlinear_input_output/structure_' + str(i + 1) + '/weights')
        with open("./Result/result_data.txt", "a") as fp:
            fp.write('nonlinear_input_output_structure_' + str(i + 1))
            fp.write(str(history.history))
            fp.write('\n')

        model = build_model_input_output(number_of_outputs=1, number_of_inputs=1, batch_size=1, struct=i+1)
        model.set_weights(weights)
        model.reset_states()
        predicted_train_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 1))
        for j in range(TEST_DATA_LENGTH):
            predicted_train_outputs[j, :] = model.predict(x_train_nonlinear_io[j, :, :])

        model.reset_states()
        predicted_test_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 1))
        for j in range(TEST_DATA_LENGTH):
            predicted_test_outputs[j, :] = model.predict(x_test_nonlinear_io[j, :, :])
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(y_train_nonlinear_io[:TEST_DATA_LENGTH, 0], 'g', label='true output', linewidth=2)
        axs[0].plot(predicted_train_outputs[:, 0], '--r', label='predicted output', linewidth=2)
        axs[0].set_title('train: x_1')
        axs[0].legend()
        axs[1].plot(y_test_nonlinear_io[:TEST_DATA_LENGTH, 0], '--g', label='true output', linewidth=2)
        axs[1].plot(predicted_test_outputs[:, 0], '--r', label='predicted output', linewidth=2)
        axs[1].set_title('train: x_2')
        axs[1].legend()
        plt.savefig('./Result/figures/nonlinear_input_output_structure_' + str(i + 1))

        ''' Nonlinear system noise. Input -> State Space Vector '''
        model = build_model_input_output(number_of_outputs=1, number_of_inputs=1, batch_size=BATCH_SIZE, struct=i+1)
        model.summary()
        x_train = tf.ragged.constant([x_train_nonlinear_io[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
        y_train = np.copy(y_train_nonlinear_io[0:TRAIN_DATA_LENGTH, :])
        y_train += np.random.normal(0, 0.5, size=(TRAIN_DATA_LENGTH, 1))
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        weights = model.get_weights()
        model.save_weights('./Result/models/nonlinear_input_output_noise/structure_' + str(i + 1) + '/weights')
        with open("./Result/result_data.txt", "a") as fp:
            fp.write('nonlinear_input_output_noise_structure_' + str(i + 1))
            fp.write(str(history.history))
            fp.write('\n')

        model = build_model_input_output(number_of_outputs=1, number_of_inputs=1, batch_size=1, struct=i+1)
        model.set_weights(weights)
        model.reset_states()
        predicted_train_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 1))
        for j in range(TEST_DATA_LENGTH):
            predicted_train_outputs[j, :] = model.predict(x_train_nonlinear_io[j, :, :])

        model.reset_states()
        predicted_test_outputs = np.zeros(shape=(TEST_DATA_LENGTH, 1))
        for j in range(TEST_DATA_LENGTH):
            predicted_test_outputs[j, :] = model.predict(x_test_nonlinear_io[j, :, :])
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(y_train_nonlinear_io[:TEST_DATA_LENGTH, 0], 'g', label='true output', linewidth=2)
        axs[0].plot(predicted_test_outputs[:, 0], '--r', label='predicted output', linewidth=2)
        axs[0].set_title('train: x_1')
        axs[0].legend()
        axs[1].plot(y_test_nonlinear_io[:TEST_DATA_LENGTH, 0], '--g', label='true output', linewidth=2)
        axs[1].plot(predicted_test_outputs[:, 0], '--r', label='predicted output', linewidth=2)
        axs[1].set_title('train: x_2')
        axs[1].legend()
        plt.savefig('./Result/figures/nonlinear_input_output_noise_structure_' + str(i + 1))
        plt.close('all')
