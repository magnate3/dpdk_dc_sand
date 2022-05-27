import pycuda.autoinit
import pycuda.gpuarray as cua
from pyvkfft.fft import fftn
import numpy as np

# d0 = cua.to_gpu(np.random.uniform(0,1,(200,200)).astype(np.complex64))
np_data = np.random.uniform(0,1,(200,200)).astype(np.complex64)
np_fft = np.fft.fftn(np_data)

# d0 = cua.to_gpu(np.random.uniform(0,1,(200,200)).astype(np.float32))
d0 = cua.to_gpu(np_data)

# This will compute the fft to a new GPU array
d1 = fftn(d0)

# An in-place transform can also be done by specifying the destination
d0 = fftn(d0, d0)

# Or an out-of-place transform to an existing array (the destination array is always returned)
d1 = fftn(d0, d1)

a = 1