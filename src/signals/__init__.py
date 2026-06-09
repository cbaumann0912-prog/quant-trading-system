import numpy as np
M = np.array([[3.0, 1.0], [1.0, 2.0]])
U, s, Vt = np.linalg.svd(M)
print(U.shape, s.shape, Vt.shape)
print(s)