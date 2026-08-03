import numpy as np

matrixA = np.array([[1, 0],
                    [0, 1]])
matrixB = np.array([[0, 0],
                    [0, 0],
                    [2, 0]])

# (m, n) x (n, p) only if n == p and final product is (m, p)

rowA, colA = matrixA.shape
rowB, colB = matrixB.shape

if colA == rowB:
    product = matrixA @ matrixB
    print(f" product: \n{product}, has dimensions: {product.shape}")
else:
    print(f"Matrix multiplication not possible as colA({colA}) != rowB({rowB})")



"""
    output = input @ W + b
"""
