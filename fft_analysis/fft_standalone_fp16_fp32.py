# Pinned memory based on example: https://gist.github.com/sjperkins/d9e6db1b2d6038febb72
# FFT (FP16) Example: https://docs.cupy.dev/en/stable/user_guide/fft.html
from cmath import pi
import pycuda.driver as cuda
import resource
import numpy as np
from time import process_time
from skcuda import cufft as cf
import pycuda.autoinit
from pycuda import gpuarray
import matplotlib.pyplot as plt
import cupy as cp

N = 2**18
shape = (1, 1, N)  # input array shape

def _pack_real_to_complex(data_in):
    data_out = cp.zeros((shape[0], shape[1], 2*shape[2]), dtype=np.float16)
    r = 0
    for n in range(shape[2]):
        data_out[0][0][r] = data_in[0][0][n]
        r +=2
    return data_out

def generate_data(src, scale):
    if src == 'wgn':
        return _generate_data_wgn(scale)
    elif src == 'cw':
        return _generate_data_cw(scale)
    elif src == 'const':
        return _generate_constant(scale)


def _generate_data_wgn(scale):
    # Option 1: Generate WGN
    # Store the input/output arrays as fp16 arrays twice as long, as complex32 is not yet available
    # Additonal note: CuPy is expecting the data as complex. To input real data create array as R+j0.

    # Set seed for repeatability
    cp.random.seed(1)

    # Option 1: Random complex input
    # a_in = 0.005*cp.random.random((shape[0], shape[1], 2*shape[2])).astype(cp.float16)

    # Option 2: Real input with complex formatting. Generate N random samples and either use as 
    # a N-valued real array or create a complex-valued array with imag as zero (0j).
    a_in = scale*cp.random.random((shape[0], shape[1], shape[2])).astype(cp.float16)
    # a_real = cp.zeros((shape[0], shape[1], 2*shape[2]), dtype=np.float16)
    a_real = _pack_real_to_complex(a_in)

    return (a_in, a_real)

def _generate_data_cw(scale):
    # Option 2: Generate CW
    # Store the input/output arrays as fp16 arrays twice as long, as complex32 is not yet available
    # Additonal note: CuPy is expecting the data as complex. To input real data create array as R+j0.

    # Real input with complex formatting. Generate N random samples and either use as 
    # a N-valued real array or create a complex-valued array with imag as zero (0j).
    # a_in = 0.005*cp.random.random((shape[0], shape[1], shape[2])).astype(cp.float16)
    # lin = cp.linspace(0,1,shape[2])
    # a_in = scale*cp.cos(2*pi*lin)

    f = shape[2]/512
    in_array = np.linspace(-(f*np.pi), f*np.pi, shape[2])
    a_in = scale*np.cos(in_array)
    a_in = a_in.reshape(shape)


    # plt.figure(1)
    # plt.plot(a_in[0][0])
    # plt.show()

    a_real = _pack_real_to_complex(a_in)

    # return (0, 1)
    return (a_in, a_real)

def _generate_constant(scale):
    # Option 3: Static vector array 
    vec_len = shape[2]
    d_first_real = scale*cp.ones((1,), dtype=np.float16)
    d_second_real = cp.zeros((vec_len-1,), dtype=np.float16)
    a_in = cp.concatenate([d_first_real, d_second_real])
    a_in = a_in.reshape(shape)
    a_real = _pack_real_to_complex(a_in)
    return (a_in, a_real)

def _fft_cpu(a_real):
    # ---- FFT with NumPy ----
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

    return out_np[0][0][0:int(N/2)]

def _fft_fp16_gpu(a_real):
    idtype = odtype = edtype = 'E'  # = numpy.complex32 in the future

    # --- FFT: FP16 ---
    # Creat host array in the same shape as the complex formatted input. 
    # Note: the output is half the length as it's a real input so only half 
    # spectrum output due to symmetry.
    out_fp16 = cp.empty_like(a_real)

    # FFT plan with cuFFT
    plan = cp.cuda.cufft.XtPlanNd(shape[1:],
                                shape[1:], 1, shape[1]*shape[2], idtype,
                                shape[1:], 1, shape[1]*shape[2], odtype,
                                shape[0], edtype,
                                order='C', last_axis=-1, last_size=None)

    # Run FFT plan
    plan.fft(a_real, out_fp16, cp.cuda.cufft.CUFFT_FORWARD)

    # Convert FP16 results to complex64 for comparision
    gpu_out_fp16 = out_fp16.get()
    temp = cp.asnumpy(gpu_out_fp16).astype(np.float32)
    gpu_out_cmplx64 = temp.view(np.complex64)
    return gpu_out_cmplx64[0][0][0:int(N/2)]

def _fft_gpu_fp32(a_in):

    # ---- FFT with FP32 GPU (PyCUDA) ----
    BATCH = np.int32(1)

    # Get input data. This will be a R2C FWD transform so we only want real-valued array.
    # data = a_in.get()
    # data = cp.asnumpy(data).astype(np.float32)
    # data = data.view(np.float32)

    data = cp.asnumpy(a_in).astype(np.float32)
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
    return c_pin[0:int(N/2)]

def display_results(fft_cpu_out, fft_gpu_fp32_out, fft_gpu_fp16_out):
    # plt.figure()
    # plt.plot(10*np.log10(np.power(np.abs(fft_cpu_out),2)))
    # plt.plot(10*np.log10(np.power(np.abs(fft_gpu_fp32_out),2)))
    # plt.plot(10*np.log10(np.power(np.abs(fft_gpu_fp16_out),2)))
    # plt.show()

    plt.figure()
    plt.plot(10*np.log10(np.power(np.abs(fft_cpu_out),2)))
    plt.figure()
    plt.plot(10*np.log10(np.power(np.abs(fft_gpu_fp32_out),2)))
    plt.figure()
    plt.plot(10*np.log10(np.power(np.abs(fft_gpu_fp16_out),2)))
    plt.show()

def analyse_results(fft_cpu_out, fft_gpu_fp32_out, fft_gpu_fp16_out):
    # Compute MSE for CPU, GPU(FP32) and GPU(FP16)

    # CPU vs GPU FP16
    cpu_gpu_fp16_diff = fft_cpu_out - fft_gpu_fp16_out
    mse = np.sum(np.power(cpu_gpu_fp16_diff,2))/len(cpu_gpu_fp16_diff)
    print(f'CPU vs GPU FP16 MSE: {mse}')

    # GPU FP32 vs GPU FP16
    gpu_fp32_gpu_fp16_diff = fft_gpu_fp32_out - fft_gpu_fp16_out
    mse = np.sum(np.power(gpu_fp32_gpu_fp16_diff,2))/len(gpu_fp32_gpu_fp16_diff)
    print(f'GPU FP32 vs GPU FP16 MSE: {mse}')

    # CPU vs GPU FP32
    cpu_gpu_fp32_diff = fft_cpu_out - fft_gpu_fp32_out
    mse = np.sum(np.power(cpu_gpu_fp32_diff,2))/len(cpu_gpu_fp32_diff)
    print(f'CPU vs GPU FP32 MSE: {mse}')


def main():
    # Generate data: options, 'wgn', 'cw', 'const'
    scale = 0.1
    a_in, a_real = generate_data('cw', scale)

    # Run fp16 FFT
    fft_gpu_fp16_out = _fft_fp16_gpu(a_real)

    # Run CPU(numpy) FFT
    fft_cpu_out = _fft_cpu(a_real)

    # Run fp32 FFT
    fft_gpu_fp32_out = _fft_gpu_fp32(a_in)

    # Analyse results
    analyse_results(fft_cpu_out, fft_gpu_fp32_out, fft_gpu_fp16_out)

    # Display results
    display_results(fft_cpu_out, fft_gpu_fp32_out, fft_gpu_fp16_out)

if __name__ == "__main__":
    main()