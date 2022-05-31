from re import L
import numpy as np
from matplotlib import pyplot as plt

a = np.arange(-2,2,0.01)
h = np.tanh(a)
l = 0.92*a

plt.figure()
plt.plot(a,h)
plt.plot(a,l)
plt.grid()
plt.show()
