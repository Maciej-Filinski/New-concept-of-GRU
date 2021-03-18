import numpy as np
from SystemDefinition import LinearSystem

A = np.array([[1, 0], [-1, -2]])
B = np.array([[1], [2]])
C = np.array([[2, 1]])

system = LinearSystem(A, B, C)
input_sequence = np.ones((1, 10))
output_sequence = system.linear_system_response(input_sequence, np.array([[1, 1]]))
print(output_sequence)
