import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_generator import *
from build_model import *

EPOCHS = 10
TRAIN_DATA_LENGTH = 2000
TEST_DATA_LENGTH = 100
BATCH_SIZE = 20

data = data_linear_system_input_output(number_of_sample_train=TRAIN_DATA_LENGTH,
                                       number_of_sample_test=TEST_DATA_LENGTH)
x_train_linear_io, y_train_linear_io, x_test_linear_io, y_test_linear_io = data

model = build_model_input_output(number_of_outputs=1, number_of_inputs=1, batch_size=BATCH_SIZE, struct=1)
model.summary()
x_train = tf.ragged.constant([x_train_linear_io[0:i, 0, :] for i in range(1, TRAIN_DATA_LENGTH + 1)])
y_train = np.copy(y_train_linear_io[0:TRAIN_DATA_LENGTH, :])
#y_train += np.random.normal(0, 0.5, size=(TRAIN_DATA_LENGTH, 1))
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
weights = model.get_weights()

model = build_model_input_output(number_of_outputs=1, number_of_inputs=1, batch_size=1, struct=1)
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
axs[0].set_title('train')
axs[0].legend()
axs[1].plot(y_test_linear_io[:TEST_DATA_LENGTH, 0], '--g', label='true output', linewidth=2)
axs[1].plot(predicted_test_outputs[:, 0], '--r', label='predicted output', linewidth=2)
axs[1].set_title('test')
axs[1].legend()
plt.show()