NEURAL_NETWORK_STATE_LENGTH = 2

INPUT_DNN_STRUCTURE = {
    'IDNN_output_layer': NEURAL_NETWORK_STATE_LENGTH
}

FORGET_DNN_STRUCTURE = {
    'FDNN_input_layer': 32,
    'FDNN_hidden_layer_1': 64,
    'FDNN_hidden_layer_2': 128,
    'FDNN_hidden_layer_3': 64,
    'FDNN_output_layer': NEURAL_NETWORK_STATE_LENGTH
}

OUTPUT_DNN_STRUCTURE = {
    'ODNN_input_layer': 32,
    'ODNN_output_layer': None
}
