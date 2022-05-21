import numpy as np

a = np.array([1.0,2.7,3.2])
b = np.array([1.1,2.7,3.2])

print(np.array_equal(a,b))
print(np.allclose(a,b))