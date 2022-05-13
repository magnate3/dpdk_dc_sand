#source: https://docs.cupy.dev/en/stable/user_guide/fft.html

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

shape = (1, 1, 32768)  # input array shape
# shape = (1024, 256, 256)  # input array shape

idtype = odtype = edtype = 'E'  # = numpy.complex32 in the future

# store the input/output arrays as fp16 arrays twice as long, as complex32 is not yet available
a = 0.005*cp.random.random((shape[0], shape[1], 2*shape[2])).astype(cp.float16)
out = cp.empty_like(a)

# FFT with cuFFT
plan = cp.cuda.cufft.XtPlanNd(shape[1:],
                              shape[1:], 1, shape[1]*shape[2], idtype,
                              shape[1:], 1, shape[1]*shape[2], odtype,
                              shape[0], edtype,
                              order='C', last_axis=-1, last_size=None)

plan.fft(a, out, cp.cuda.cufft.CUFFT_FORWARD)

# FFT with NumPy
a_np = cp.asnumpy(a).astype(np.float32)  # upcast
a_np = a_np.view(np.complex64)
out_np = np.fft.fftn(a_np, axes=(-2,-1))
out_np = np.ascontiguousarray(out_np).astype(np.complex64)  # downcast
out_np = out_np.view(np.float32)
out_np = out_np.astype(np.float16)

# don't worry about accruacy for now, as we probably lost a lot during casting
print('ok' if cp.mean(cp.abs(out - cp.asarray(out_np))) < 0.1 else 'not ok')

gpu_out = out.get()
diff = gpu_out[0][0] - out_np[0][0]

mse = np.sum(np.power(diff,2))/len(diff)
print(f'MSE: {mse}')

plt.figure()
plt.plot(10*np.log10(np.power(np.abs(out_np[0][0]),2)))
plt.show()

plt.figure()
plt.plot(10*np.log10(np.power(np.abs(diff),2)))
plt.show()