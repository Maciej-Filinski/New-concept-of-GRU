import numpy as np

STATE_SPACE_VECTOR_LENGTH = 2
NUMBER_OF_INPUTS = 1


def data_nonlinear_system_input_state(number_of_sample_train=2000, number_of_sample_test=100):
    train_inputs = np.random.uniform(-1, 1, size=(number_of_sample_train, 1, NUMBER_OF_INPUTS))
    train_outputs = np.zeros(shape=(number_of_sample_train + 1, STATE_SPACE_VECTOR_LENGTH))
    test_inputs = np.sin(2 * np.pi * np.array(range(number_of_sample_test)) / 50)
    test_inputs = np.reshape(test_inputs, newshape=(number_of_sample_test, 1, NUMBER_OF_INPUTS))
    test_outputs = np.zeros(shape=(number_of_sample_test + 1, STATE_SPACE_VECTOR_LENGTH))

    state_matrix = np.array([[0.7555, 0.25], [-0.1991, 0]])
    input_matrix = np.array([[-0.5], [0]])

    for i in range(1, number_of_sample_train + 1):
        if train_inputs[i - 1, 0, :] > 0:
            train_outputs[i, :] = state_matrix @ train_outputs[i - 1, :] + input_matrix @ np.sqrt(
                train_inputs[i - 1, 0, :])
        else:
            train_outputs[i, :] = state_matrix @ train_outputs[i - 1, :] + input_matrix @ train_inputs[i - 1, 0, :]

    train_outputs = train_outputs[1::, :]
    for i in range(1, number_of_sample_test + 1):
        if test_inputs[i - 1, 0, :] > 0:
            test_outputs[i, :] = state_matrix @ test_outputs[i - 1, :] + input_matrix @ np.sqrt(
                test_inputs[i - 1, 0, :])
        else:
            test_outputs[i, :] = state_matrix @ test_outputs[i - 1, :] + input_matrix @ test_inputs[i - 1, 0, :]
    test_outputs = test_outputs[1::, :]
    return train_inputs, train_outputs, test_inputs, test_outputs


def data_linear_system_input_state(number_of_sample_train=2000, number_of_sample_test=100):
    train_inputs = np.random.uniform(-1, 1, size=(number_of_sample_train, 1, NUMBER_OF_INPUTS))
    train_outputs = np.zeros(shape=(number_of_sample_train + 1, STATE_SPACE_VECTOR_LENGTH))
    test_inputs = np.sin(2 * np.pi * np.array(range(number_of_sample_test)) / 50)
    test_inputs = np.reshape(test_inputs, newshape=(number_of_sample_test, 1, NUMBER_OF_INPUTS))
    test_outputs = np.zeros(shape=(number_of_sample_test + 1, STATE_SPACE_VECTOR_LENGTH))

    state_matrix = np.array([[0.7555, 0.25], [-0.1991, 0]])
    input_matrix = np.array([[-0.5], [0]])

    for i in range(1, number_of_sample_train + 1):
        train_outputs[i, :] = state_matrix @ train_outputs[i - 1, :] + input_matrix @ train_inputs[i - 1, 0, :]

    train_outputs = train_outputs[1::, :]
    for i in range(1, number_of_sample_test + 1):
        test_outputs[i, :] = state_matrix @ test_outputs[i - 1, :] + input_matrix @ test_inputs[i - 1, 0, :]
    test_outputs = test_outputs[1::, :]
    return train_inputs, train_outputs, test_inputs, test_outputs


def data_nonlinear_system_input_output(number_of_sample_train=2000, number_of_sample_test=100):
    train_inputs = np.random.uniform(-1, 1, size=(number_of_sample_train, 1, NUMBER_OF_INPUTS))
    train_outputs = np.zeros(shape=(number_of_sample_train + 1, STATE_SPACE_VECTOR_LENGTH))
    test_inputs = 0.5 * np.sin(2 * np.pi * np.array(range(number_of_sample_test)) / 50)
    test_inputs = np.reshape(test_inputs, newshape=(number_of_sample_test, 1, NUMBER_OF_INPUTS))
    test_outputs = np.zeros(shape=(number_of_sample_test + 1, STATE_SPACE_VECTOR_LENGTH))

    state_matrix = np.array([[0.7555, 0.25], [-0.1991, 0]])
    input_matrix = np.array([[-0.5], [0]])
    output_matrix = np.array([0.6993, -0.4427])

    for i in range(1, number_of_sample_train + 1):
        if train_inputs[i - 1, 0, :] > 0:
            train_outputs[i, :] = state_matrix @ train_outputs[i - 1, :] + input_matrix @ np.sqrt(
                train_inputs[i - 1, 0, :])
        else:
            train_outputs[i, :] = state_matrix @ train_outputs[i - 1, :] + input_matrix @ train_inputs[i - 1, 0, :]

    train_outputs = train_outputs[1::, :]
    train_state = np.copy(train_outputs)
    outputs = []
    for i in range(number_of_sample_train):
        outputs.append(output_matrix @ train_outputs[i, :] + 5 * np.sin(output_matrix @ train_outputs[i, :]))
    train_outputs = np.reshape(np.array(outputs), newshape=(number_of_sample_train, 1))
    for i in range(1, number_of_sample_test + 1):
        if test_inputs[i - 1, 0, :] > 0:
            test_outputs[i, :] = state_matrix @ test_outputs[i - 1, :] + input_matrix @ np.sqrt(
                test_inputs[i - 1, 0, :])
        else:
            test_outputs[i, :] = state_matrix @ test_outputs[i - 1, :] + input_matrix @ test_inputs[i - 1, 0, :]
    test_outputs = test_outputs[1::, :]
    outputs = []
    test_state = np.copy(test_outputs)
    for i in range(number_of_sample_test):
        outputs.append(output_matrix @ test_outputs[i, :] + 5 * np.sin(output_matrix @ test_outputs[i, :]))
    test_outputs = np.reshape(np.array(outputs), newshape=(number_of_sample_test, 1))
    return train_inputs, train_outputs, test_inputs, test_outputs, (train_state, test_state)


def data_linear_system_input_output(number_of_sample_train=2000, number_of_sample_test=100):
    train_inputs = np.random.uniform(-1, 1, size=(number_of_sample_train, 1, NUMBER_OF_INPUTS))
    train_outputs = np.zeros(shape=(number_of_sample_train + 1, STATE_SPACE_VECTOR_LENGTH))
    test_inputs = np.sin(2 * np.pi * np.array(range(number_of_sample_test)) / 50)
    test_inputs = np.reshape(test_inputs, newshape=(number_of_sample_test, 1, NUMBER_OF_INPUTS))

    test_inputs = np.random.uniform(-1, 1, size=(number_of_sample_train, 1, NUMBER_OF_INPUTS))
    test_outputs = np.zeros(shape=(number_of_sample_test + 1, STATE_SPACE_VECTOR_LENGTH))

    state_matrix = np.array([[0.7555, 0.25], [-0.1991, 0]])
    input_matrix = np.array([[-0.5], [0]])
    output_matrix = np.array([0.6993, -0.4427])

    for i in range(1, number_of_sample_train + 1):
        train_outputs[i, :] = state_matrix @ train_outputs[i - 1, :] + input_matrix @ train_inputs[i - 1, 0, :]
    train_outputs = train_outputs[1::, :]
    outputs = []
    for i in range(number_of_sample_train):
        outputs.append(output_matrix @ train_outputs[i, :])
    train_outputs = np.reshape(np.array(outputs), newshape=(number_of_sample_train, 1))
    for i in range(1, number_of_sample_test + 1):
        test_outputs[i, :] = state_matrix @ test_outputs[i - 1, :] + input_matrix @ test_inputs[i - 1, 0, :]
    test_outputs = test_outputs[1::, :]
    outputs = []
    for i in range(number_of_sample_test):
        outputs.append(output_matrix @ test_outputs[i, :])
    test_outputs = np.reshape(np.array(outputs), newshape=(number_of_sample_test, 1))
    return train_inputs, train_outputs, test_inputs, test_outputs
