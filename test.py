import h5py
import numpy as np

a=[]
a.append([1,2,3])
a.append([1,2])
a = np.array(a,dtype=object)
print(type(a),a)

hf = h5py.File('test.h5','w')
hf.create_dataset('a',data=a)