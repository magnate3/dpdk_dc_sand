import numpy as np
import os

# Reference Numpy version FP32
# ----------------------------
real_data_fp32 = np.random.uniform(0,1,(1,1024)).astype(np.float32)
np_fft_fp32 = np.fft.fftn(real_data_fp32)

real_data_fp16 = real_data_fp32.astype(np.float16)
np_fft_fp16 = np.fft.fftn(real_data_fp16)

# vkFFT:
# ------
import pyopencl as cl
import pyopencl.array as cla
from pyvkfft.opencl import VkFFTApp

if 'PYOPENCL_CTX' in os.environ:
    cl_ctx = cl.create_some_context()
else:
    cl_ctx = None
    # Find the first OpenCL GPU available and use it, unless
    for p in cl.get_platforms():
        for d in p.get_devices():
            if d.type & cl.device_type.GPU == 0:
                continue
            gpu_name_real = d.name
            print("Selected OpenCL device: ", d.name)
            cl_ctx = cl.Context(devices=(d,))
            break
        if cl_ctx is not None:
            break
cq = cl.CommandQueue(cl_ctx)

# vkFFT FP32:
# -----------
d = cla.to_device(cq,real_data_fp32)
app = VkFFTApp(d.shape, d.dtype, ndim=1, r2c=True, queue=cq)
vkfft_fp32 = app.fft(d).get()


# vkFFT FP16:
# -----------
import pycuda.autoinit
import pycuda.gpuarray as cua
import pycuda.driver as cuda
from pyvkfft.cuda import VkFFTApp as VkFFTAppCu

stream = cuda.Stream()
d0 = cua.to_gpu(real_data_fp16)
cuda.Context.synchronize()
app = VkFFTAppCu(real_data_fp16.shape, real_data_fp16.dtype, ndim=1, r2c=True, queue=stream, inplace=False)
dst_size = d0.size // d0.shape[-1] * (d0.shape[-1] // 2 + 1)
out_shape = (1,dst_size)
npzeros = np.zeros(out_shape)
vkout = cua.to_gpu(npzeros)
vkout = app.fft(d0,vkout)
cuda.Context.synchronize()
vkfft_fp16 = vkout.get()

print(f'NP FP32 first 5 entries:{np_fft_fp32[0][0:5]}')
print(f'NP FP16 first 5 entries:{np_fft_fp16[0][0:5]}')
print(f'vkFFT FP32 first 5 entries:{vkfft_fp32[0][0:5]}')
print(f'vkFFT FP16 first 5 entries:{vkfft_fp16[0][0:5]}')