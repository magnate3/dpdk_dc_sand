# Note:
# Pinned memory based on example: https://gist.github.com/sjperkins/d9e6db1b2d6038febb72
# FFT (FP16) Example: https://docs.cupy.dev/en/stable/user_guide/fft.html

import os
import config
import numpy as np
import cupy as cp
import resource
import pyopencl as cl
import pyopencl.array as cla
from skcuda import cufft as cf
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as cua
from pyvkfft.opencl import VkFFTApp

N = config.N
shape = config.shape

def fft_gpu(input_real_fp64, input_cmplx_interleave_fp64):
    def _vkfft_gpu(input_real_fp64, input_cmplx_interleave_fp64):
        # Find an opencl device
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

        # vkFFT FP32 input, FP32 out
        input_fp32 = input_real_fp64[0].astype(np.float32)

        d = cla.to_device(cq,input_fp32)
        app = VkFFTApp(d.shape, d.dtype, ndim=1, r2c=True, queue=cq)
        vkfft_fp_32 = app.fft(d)

        # plt.figure(1)
        # fft_power_spec = np.power(np.abs(vkfft_fp_32.get()[0]),2)
        # number_samples = len(fft_power_spec)*2
        # num_steps = 8
        # plt.plot(10*np.log10(fft_power_spec))
        # plt.title(f'FFT: GPU(vkFFT FP32) - {round(2048*1712e6/number_samples/1e6)}MHz')
        # # plt.title('GPU(vkFFT FP32)')
        # labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
        # labels = labels.round(0)
        # plt.xticks(np.arange(0, (len(fft_power_spec)+len(fft_power_spec)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
        # plt.xlabel('Frequency (MHz)')
        # plt.ylabel('dB')
        # plt.show()

        return vkfft_fp_32.get()[0]

    def _fft_fp16_gpu(input_cmplx_interleave_fp64):

        # Convert to cupy array and cast to fp16
        input_cmplx_int_fp16 = cp.asarray(input_cmplx_interleave_fp64.astype(cp.float16))

        idtype = odtype = edtype = 'E'  # = numpy.complex32 in the future

        # --- FFT: FP16 ---
        # Creat host array in the same shape as the complex formatted input. 
        # Note: the output is half the length as it's a real input so only half 
        # spectrum output due to symmetry.
        out_fp16 = cp.empty_like(input_cmplx_int_fp16)

        # FFT plan with cuFFT
        plan = cp.cuda.cufft.XtPlanNd(shape[1:],
                                    shape[1:], 1, shape[1]*shape[2], idtype,
                                    shape[1:], 1, shape[1]*shape[2], odtype,
                                    shape[0], edtype,
                                    order='C', last_axis=-1, last_size=None)

        # Run FFT plan
        plan.fft(input_cmplx_int_fp16, out_fp16, cp.cuda.cufft.CUFFT_FORWARD)

        # Convert FP16 results to complex64 for comparision
        gpu_out_fp16 = out_fp16.get()
        temp = cp.asnumpy(gpu_out_fp16).astype(np.float32)
        gpu_out_cmplx64 = temp.view(np.complex64)
        return gpu_out_cmplx64[0][0][0:int(N/2)]

    def _fft_gpu_fp32(input_real_fp64):

        # Get input data. This will be a R2C FWD transform so we only want real-valued array.
        # Cast input to FP32.
        input_real_fp32 = input_real_fp64.astype(np.float32)

        # ---- FFT with FP32 GPU (PyCUDA) ----
        BATCH = np.int32(1)

        # Create a stream
        stream = cuda.Stream()

        a = cuda.aligned_empty((shape[2],), dtype=np.float32, alignment=resource.getpagesize())
        c = cuda.aligned_empty((int(shape[2]/2+1),), dtype=np.complex64, alignment=resource.getpagesize())

        # Assign input data.
        a[:] = input_real_fp32[0][0]

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
        return c_pin[0:int(N/2)]

    # Run fp16 FFT
    fft_gpu_fp16_out = _fft_fp16_gpu(input_cmplx_interleave_fp64)
    
    # Run fp32 FFT
    fft_gpu_fp32_out = _fft_gpu_fp32(input_real_fp64)

    # Run VKFFT
    fft_gpu_vkfft = _vkfft_gpu(input_real_fp64, input_cmplx_interleave_fp64)

    return (fft_gpu_fp32_out, fft_gpu_fp16_out, fft_gpu_vkfft)
