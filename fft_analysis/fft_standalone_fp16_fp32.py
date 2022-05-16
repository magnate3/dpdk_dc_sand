#Pinned memory based on example: https://gist.github.com/sjperkins/d9e6db1b2d6038febb72
import pycuda.driver as cuda
import resource
import numpy as np
from time import process_time
from skcuda import cufft as cf
import pycuda.autoinit
from pycuda import gpuarray
import matplotlib.pyplot as plt
# from IPython import embed
# from pycuda.compiler import SourceModule
# import pycuda.tools as tools


# --- FP16 ---
#source: https://docs.cupy.dev/en/stable/user_guide/fft.html
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

shape = (1, 1, 8)  # input array shape
# shape = (1, 8)  # input array shape

idtype = odtype = edtype = 'E'  # = numpy.complex32 in the future

cp.random.seed(1)
# store the input/output arrays as fp16 arrays twice as long, as complex32 is not yet available
# a_in = 0.005*cp.random.random((shape[0], shape[1], 2*shape[2])).astype(cp.float16)

a_in = 0.005*cp.random.random((shape[0], shape[1], shape[2])).astype(cp.float16)
a_real = cp.zeros((shape[0], shape[1], 2*shape[2]), dtype=np.float16)

q = 0
for i in range(shape[2]):
    a_real[0][0][q] = a_in[0][0][i]
    q +=2

# vec_len = shape[2]
# d_first_real = cp.ones((1,), dtype=np.float16)
# d_second_real = cp.zeros((vec_len-1,), dtype=np.float16)
# b = cp.concatenate([d_first_real, d_second_real])
# b = b.reshape(shape)
# a_in = b

out_fp16 = cp.empty_like(a_in)

# FFT with cuFFT
plan = cp.cuda.cufft.XtPlanNd(shape[1:],
                              shape[1:], 1, shape[1]*shape[2], idtype,
                              shape[1:], 1, shape[1]*shape[2], odtype,
                              shape[0], edtype,
                              order='C', last_axis=-1, last_size=None)

plan.fft(a_real, out_fp16, cp.cuda.cufft.CUFFT_FORWARD)

# Convert FP16 results to complex64 for comparision
gpu_out_fp16 = out_fp16.get()
temp = cp.asnumpy(gpu_out_fp16).astype(np.float32)
gpu_out_cmplx64 = temp.view(np.complex64)

# FFT with NumPy
a_np = cp.asnumpy(a_real).astype(np.float32)  # upcast
a_np = a_np.view(np.complex64)

out_np = np.fft.fftn(a_np)
# out_np = np.fft.fftn(a_np, axes=(-2,-1))
# out_np_temp = out_np.astype(np.complex64)
# out_np = np.ascontiguousarray(out_np).astype(np.complex64)  # downcast
# out_np = out_np.view(np.float32)
# out_np = out_np.astype(np.float16)

# don't worry about accruacy for now, as we probably lost a lot during casting
# print('ok' if cp.mean(cp.abs(out_fp16 - cp.asarray(out_np))) < 0.1 else 'not ok')


# diff = gpu_out[0][0] - out_np[0][0]
# mse = np.sum(np.power(diff,2))/len(diff)
# print(f'MSE: {mse}')

# plt.figure()
# plt.plot(10*np.log10(np.power(np.abs(out_np[0][0]),2)))
# plt.show()

# plt.figure()
# plt.plot(10*np.log10(np.power(np.abs(diff),2)))
# plt.show()


# --- FP32 ---
BATCH = np.int32(1)

# Get input data. This will be a R2C FWD transform so we only want real-valued array.
data = a_in.get()
data = cp.asnumpy(data).astype(np.float32)
data = data.view(np.float32)

# speedup_log = []
# speedup_log_with_transfer = []
# cpu_time_log = []
# gpu_time_log = []
# gpu_time_transfer_log = []

# Create a stream
stream = cuda.Stream()

a = cuda.aligned_empty((shape[2],), dtype=np.float32, alignment=resource.getpagesize())
c = cuda.aligned_empty((int(shape[2]/2+1),), dtype=np.complex64, alignment=resource.getpagesize())

# Assign input data.
a[:] = data[0][0]

# Allocate number of bytes required by A on the GPU
a_gpu = cuda.mem_alloc(a.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Pin the host memory
a_pin = cuda.register_host_memory(a)
c_pin = cuda.register_host_memory(c)

# Make FFT plan
plan = cf.cufftPlan1d(len(a), cf.CUFFT_R2C, BATCH)

# --- Memcopy from host to device
# Asynchronously copy pinned array data to the gpu array
cuda.memcpy_htod_async(a_gpu, a_pin, stream)
# Execute FFT plan
res = cf.cufftExecR2C(plan, int(a_gpu), int(c_gpu))

# Transfer gpu array data to pinned memory
cuda.memcpy_dtoh_async(c_pin, c_gpu, stream)

# Synchronize
cuda.Context.synchronize()

# diff = gpu_out[0][0] - c_pin
# mse = np.sum(np.power(diff,2))/len(diff)
# print(f'MSE: {mse}')

plt.figure()
plt.plot(10*np.log10(np.power(np.abs(c_pin),2)))
plt.plot(10*np.log10(np.power(np.abs(gpu_out[0][0]),2)))
plt.show()