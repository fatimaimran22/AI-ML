"""
    Eigenvectors are special vectors that, when multiplied by a matrix, do not change direction. They only get stretched, shrunk, or flipped.
    Av = λv

    The eigenvalue tells how much the vector is stretched or shrunk.
    Positive λ → same direction
    Negative λ → opposite direction (flipped)

    PCA (Principal Component Analysis)--------> Reduce dimensions while preserving as much information (variance) as possible.

    1)First build Covariance Matrix ----->The covariance matrix summarizes how all the features vary together. 
    (e.g how much does x varies, how much does y varies and how much does x and y varies together.)

    2) Find Eigenvectors → tell you which directions are important.
            Eigenvalues → tell you how much variance (stretch) exists along each direction.
"""

import numpy as np

a = np.array([[1, 2],   
              [2, 4],
              [6, 5]])

cov = np.cov(a.T) # ----> cov takes rows as features so we transpose it
eigenvals, eigenvectors = np.linalg.eig(cov)
print(f"Eigen Values: {eigenvals}")
order = np.argsort(eigenvals)[::-1]     #return array of indexes in sorted way 
#(::-1 means reverse it) as args sort return ascending order
print(f"Sorted Order Index: {order}")
print("Explained variance ratio:", (eigenvals[order] / eigenvals.sum()))
print("Explained variance percentage:", (eigenvals[order] / eigenvals.sum())* 100)
print("PC1 (Principal Component 1) captures 95% of all the variation (information) in your data. PC2 captures only 4.9%. if PC1 retains 95% keep PC1 and discard PC2")
print("\n\n\n")



""" UNIVERSITY EXAMPLE
| Height | Weight |
| ------ | ------ |
| 150    | 45     |
| 155    | 48     |
| 160    | 52     |
| 165    | 56     |
| 170    | 60     |
| 175    | 65     |
"""

data = np.array([[150, 45],
                 [155, 48],
                 [160, 52],
                 [165, 56],
                 [170, 60],
                 [175, 65]])


cov_matrix = np.cov(data.T)
print(f"Covariance Matrix:\n {cov_matrix}")

eigenvals,eigenvectors = np.linalg.eig(cov_matrix)
print(f"Eigen Vectors:\n {eigenvectors}")
print(f"Eigen Values: {eigenvals}")
order = np.argsort(eigenvals)[::-1]
variance = eigenvals[order] / eigenvals.sum() * 100

print(f"PC1 explains {variance[0]:.2f}% variance")
print(f"PC2 explains {variance[1]:.2f}% variance")
if variance[0] > 95:
    print("PC1 (this new dimension) alone is enough.")
else:
    print("Keep both principal components.")

# Sort eigenvectors by descending eigenvalues
sorted_vectors = eigenvectors[:, order] #np array [rows, cols]-----> (:) means Take all rows, order [1, 0] first keep second col then first

# Keep only PC1
pc1 = sorted_vectors[:, 0]

# Project the data onto PC1
reduced_data = data @ pc1

print(reduced_data)


"""
PCA finds new directions

So instead of using 1,000 features, you use only 50 principal components.

Benefits:

Faster training
Less memory
Less noise
Reduced risk of overfitting

Original Data
      │
      ▼
Compute Covariance Matrix
      │
      ▼
Find Eigenvalues & Eigenvectors
      │
      ▼
Sort by Largest Eigenvalues
      │
      ▼
Keep Top k Principal Components
      │
      ▼
Project Original Data onto Those Components
      │
      ▼
Use the Reduced Data for Machine Learning
"""

