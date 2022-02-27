import numpy as np
from matplotlib import pyplot as plt
from mode_decomposition import *

a = np.arange(20)
a = np.reshape(a,(-1,5))
print(a.shape)

modes,lam = POD(a)