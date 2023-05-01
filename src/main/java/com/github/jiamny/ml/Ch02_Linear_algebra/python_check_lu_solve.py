import numpy as np

from scipy.linalg import lu_factor, lu_solve

A = np.array([[0.7220180, 0.07121225, 0.6881997],
        [-0.2648886, -0.89044952, 0.3700456],
        [-0.6391588, 0.44947578, 0.6240573]])

b = np.array([1.0, 2.0, 3.0])

lu, piv = lu_factor(A)

x = lu_solve((lu, piv), b)
print(x)

# [-1.72523568 -0.36125924  3.30046249]
