from NewGRU import NewGRU
from tensorflow import keras
from tensorflow.keras.layers import RNN
import tensorflow as tf
from SystemDefinition import LinearSystem
import matplotlib.pyplot as plt
import numpy as np

state_length = 10
time_step = 32
numbers_of_system_input = 1
batch_size = 2

cell = NewGRU(state_length)
rnn = keras.layers.RNN(cell)

input_1 = keras.Input((time_step, numbers_of_system_input))
outputs = rnn(input_1)
model = keras.models.Model(input_1, outputs)

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model.summary()
print(model(tf.random.uniform(shape=(batch_size, time_step, numbers_of_system_input))))
