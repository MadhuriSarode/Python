# Numpy

import numpy as np

original_vector = (np.random.random(20))  # Random vector of size 20 having only floating numbers the range 1-20
print("original_vector=")
print(original_vector)

reshaped_vector = (original_vector.reshape(4, 5))
print("\nreshaped_vector =")
print(reshaped_vector)

max_values = (np.max(reshaped_vector, axis=1))  # Extracting max values
max_val_reshaped_matrix = max_values.reshape(-1, 1)  # Reshaping the matrix into a single column of max values
print("\n max val reshaped matrix=")
print(max_val_reshaped_matrix)

final_matrix = np.where(reshaped_vector == max_val_reshaped_matrix, 0, reshaped_vector)     # Replacing the maximum value with 0
print("\n Updated matrix is below:Using Solution1  \n")
print("\n Final matrix =")
print(final_matrix)  # Printing final updated matrix

b = reshaped_vector
b[np.arange(len(reshaped_vector)), reshaped_vector.argmax(1)] = 0
print("\n Updated matrix is below:Using Solution2  \n")
print("Final matrix = ", b)
