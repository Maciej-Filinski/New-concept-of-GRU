import json
import os


NEURAL_NETWORK_STATE_LENGTH = 5
FILE_NAME = 'structure_1.json'
STRUCT = {'state_length': NEURAL_NETWORK_STATE_LENGTH,
          'candidate_dnn': {'IDNN_input_layer': 32,
                            'IDNN_hidden_layer_1': 64,
                            'IDNN_hidden_layer_2': 32,
                            'IDNN_output_layer': NEURAL_NETWORK_STATE_LENGTH},
          'forget_dnn': {'FDNN_input_layer': 32,
                         'FDNN_hidden_layer_1': 64,
                         'FDNN_hidden_layer_2': 32,
                         'FDNN_output_layer': NEURAL_NETWORK_STATE_LENGTH},
          'output_dnn': {'ODNN_input_layer': 32,
                         'ODNN_hidden_layer_1': 64,
                         'ODNN_hidden_layer_2': 32,
                         'ODNN_output_layer': None},
          'forget_dnn_enable': True,
          'output_dnn_enable': True
          }


if __name__ == '__main__':

    file_path = os.path.join(os.path.abspath('../neural_network_structure'), FILE_NAME)
    if os.path.exists(file_path) is True:
        decision = input('Overwrite the file? yes/no: ')
    else:
        decision = 'yes'
    if decision.lower() == 'yes':
        with open(file_path, 'w') as file:
            json.dump(STRUCT, file, indent=4)
        print('The data has been saved.')
    else:
        print('The data has not been saved.')
