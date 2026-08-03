import numpy as np

scalar = 5.0
vector = np.array([1, 2, 2, 4])     #shape (4,) -> 1D
matrix = np.array([[1, 2, 5, 6], 
                   [1, 6, 3, 7]])     #shape (2,4)  -> 2D
tensor = np.random.rand(32, 28, 28, 3)     #shape (32, 28, 28, 3) -> 4D
#(batch, height, width, channels)   -> size of each dimension

print(f"Vector: {vector.shape}")
print(f"Matrix: {matrix.shape}")
print(f"Tensor: {tensor.shape}")