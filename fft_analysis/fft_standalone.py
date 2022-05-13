#Pinned memory based on example: https://gist.github.com/sjperkins/d9e6db1b2d6038febb72
import pycuda.driver as cuda
import resource
import numpy as np
from time import process_time
from skcuda import cufft as cf
import pycuda.autoinit
from pycuda import gpuarray
import matplotlib.pyplot as plt
from IPython import embed
from pycuda.compiler import SourceModule
import pycuda.tools as tools


# import numpy as np
# import pycuda.autoinit
# from pycuda.elementwise import ElementwiseKernel as CU_ElK
# import pycuda.gpuarray as cua
# cu_options = ['-use_fast_math', '-D__CUDA_NO_HALF_OPERATORS__', 
# '-D__CUDA_NO_HALF2_OPERATORS__']

# testk = CU_ElK(name='testk', operation="d[i] *= 2", preamble='#include <cuda_fp16.h>',
#                options=cu_options, arguments="float *d")
# cu_d = cua.empty(128, dtype=np.float16)
# test = testk(cu_d)

num_iter = 200
BATCH = np.int32(1)
vec_len = np.power(2,16)

# Generate complex data vector
# d_first_cmplx = ((np.ones((1,), dtype=np.int32))+ 1j*(np.zeros((1,), dtype=np.int32))).astype(np.complex64)
# d_second_cmplx = ((np.zeros((vec_len-1,), dtype=np.int32))+ 1j*(np.zeros((vec_len-1,), dtype=np.int32))).astype(np.complex64) 

d_first_real = (np.ones((1,), dtype=np.float32))
d_second_real = (np.zeros((vec_len-1,), dtype=np.float32))

# Concat the two pieces to form full vector
# data_cmplx = np.concatenate([d_first_cmplx, d_second_cmplx])
data_real = np.concatenate([d_first_real, d_second_real])
# data = data_cmplx
# data = data_real

speedup_log = []
speedup_log_with_transfer = []
cpu_time_log = []
gpu_time_log = []
gpu_time_transfer_log = []

# create two timers so we can speed-test each approach
start = cuda.Event()
end = cuda.Event()
start_kernel_time = cuda.Event()
end_kernel_time = cuda.Event()

# Create a stream
stream = cuda.Stream()

a = cuda.aligned_empty((len(data_real),), dtype=np.float32, alignment=resource.getpagesize())
c = cuda.aligned_empty((int(len(data_real)/2+1),), dtype=np.complex64, alignment=resource.getpagesize())
# Note: for R2C transform the input data length is N, but the output is N/2+1. 
# See: https://docs.nvidia.com/cuda/cufft/index.html#data-layout
# https://forums.developer.nvidia.com/t/2d-cufft-wrong-result/37671

a[:] = data_real
# print(a.nbytes)

# Allocate number of bytes required by A on the GPU
a_gpu = cuda.mem_alloc(a.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Pin the host memory
a_pin = cuda.register_host_memory(a)
c_pin = cuda.register_host_memory(c)

# Uncomment, this works
#a_pin = drv.pagelocked_empty(shape=shape, dtype=dtype)
#a_pin[:] = 100

assert np.all(a_pin == a)
assert np.all(c_pin == c)

# Make FFT plan
# plan = cf.cufftPlan1d(vec_len, cf.CUFFT_C2C, BATCH)
plan = cf.cufftPlan1d(vec_len, cf.CUFFT_R2C, BATCH)

# plan = cf.cufftPlanMany(1, a.ctypes.data, None, 1, 0, None, 1, 0, cf.CUFFT_R2C, BATCH)
# plan = cf.cufftXtMakePlanMany(1, a.ctypes.data, None, 1, 0, None, 1, 0, cf.CUFFT_R2C, BATCH)

# start timing using events
start.record() 
# --- Memcopy from host to device
# Asynchronously copy pinned array data to the gpu array
cuda.memcpy_htod_async(a_gpu, a_pin, stream)
# Execute FFT plan
start_kernel_time.record()
# res = cf.cufftExecC2C(plan, int(a_gpu), int(c_gpu), cf.CUFFT_FORWARD)
res = cf.cufftExecR2C(plan, int(a_gpu), int(c_gpu))
end_kernel_time.record()

# Transfer gpu array data to pinned memory
cuda.memcpy_dtoh_async(c_pin, c_gpu, stream)
# end timing using events
end.record() 

# Synchronize
cuda.Context.synchronize()
#end_kernel_time.synchronize()
gpu_time = start_kernel_time.time_till(end_kernel_time)*1e-3
# calculate the run length
gpu_time_transfer = start.time_till(end)*1e-3

# calculate and time NumPy FFT
t1 = process_time()
dataFft = np.fft.fft(data_real)
t2 = process_time()
cpu_time = t2-t1

print('\nCPU NumPy time is: ',cpu_time)
print('\nGPU time (with transfer) is: ',gpu_time_transfer)
print("GPU Speedup (with transfer) is:",cpu_time/gpu_time_transfer)
speedup_log_with_transfer.append(cpu_time/gpu_time_transfer)  
print('\nGPU (kernel) time is: ',gpu_time)
print("GPU (kernel) Speedup is:",cpu_time/gpu_time)
print('')
speedup_log.append(cpu_time/gpu_time)

cpu_time_log.append(cpu_time)
gpu_time_log.append(gpu_time)
gpu_time_transfer_log.append(gpu_time_transfer)

plt.figure()
plt.plot(np.abs(c_pin))
plt.show()

# print(speedup_log)
# print('')
# print(speedup_log_with_transfer)

# fig = plt.figure()
# fig.subplots_adjust(top=3.5)
# ax1 = fig.add_subplot(311)
# ax1.set_ylabel('Execution Time(s)')
# ax1.set_xlabel('Iteration')
# ax1.set_title('CPU Execution Time')
# ax1.plot(cpu_time_log)

# ax2 = fig.add_subplot(312)
# ax2.set_ylabel('Execution Time(s)')
# ax2.set_xlabel('Iteration')
# ax2.set_title('GPU Execution Time (Kernel)')
# ax2.plot(gpu_time_log)

# ax3 = fig.add_subplot(313)
# ax3.set_ylabel('Execution Time(s)')
# ax3.set_xlabel('Iteration')
# ax3.set_title('GPU Execution Time (Kernel + Transfer)')
# ax3.plot(gpu_time_transfer_log)

# fig = plt.figure()
# fig.subplots_adjust(top=2.25)
# ax1 = fig.add_subplot(211)
# ax1.set_ylabel('Speedup(x)')
# ax1.set_xlabel('Iteration')
# ax1.set_title('GPU Speedup (Kernel)')
# ax1.plot(speedup_log)

# ax2 = fig.add_subplot(212)
# ax2.set_ylabel('Speedup(x)')
# ax2.set_xlabel('Iteration')
# ax2.set_title('GPU Speedup(Kernel + Transfer)')
# ax2.plot(speedup_log_with_transfer)

# diff = dataFft - c_pin

# plt.figure()
# plt.ion()
# plt.clf()
# plt.plot(diff)
# plt.show()

# print(dataFft)

a_gpu.free()
c_gpu.free()
cf.cufftDestroy(plan)


# free, total = cuda.mem_get_info()
# print('Current Free Memory(%)')
# print((free/float(total))*100)