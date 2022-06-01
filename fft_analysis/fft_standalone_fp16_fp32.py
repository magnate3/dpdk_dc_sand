# Pinned memory based on example: https://gist.github.com/sjperkins/d9e6db1b2d6038febb72
# FFT (FP16) Example: https://docs.cupy.dev/en/stable/user_guide/fft.html
from cmath import log10, nan, pi
from fileinput import filename
import pycuda.driver as cuda
import resource
import numpy as np
from time import process_time
from skcuda import cufft as cf
import pycuda.autoinit
from pycuda import gpuarray
import matplotlib.pyplot as plt
import cupy as cp
import os
import h5py

import pycuda.autoinit
import pycuda.gpuarray as cua
from pyvkfft.fft import fftn

from scipy.misc import ascent
import pyopencl as cl
import pyopencl.array as cla
from pyvkfft.opencl import VkFFTApp

N = 2**16
shape = (1, 1, N)  # input array shape

def generate_data(src, scale=0.1):
    
    def _pack_real_to_complex_interleave(data_in):
        data_out = np.zeros((shape[0], shape[1], 2*shape[2]), dtype=np.float64)
        r = 0
        for n in range(shape[2]):
            data_out[0][0][r] = data_in[0][0][n]
            r +=2
        return data_out

    def _generate_data_wgn(scale):
        # Option 1: Generate WGN
        # Store the input/output arrays as fp16 arrays twice as long, as complex32 is not yet available
        # Additonal note: CuPy is expecting the data as complex. To input real data create array as R+j0.

        # Set seed for repeatability
        cp.random.seed(1)

        # Real input with complex formatting. Generate N random samples and either use as 
        # a N-valued real array or create a complex-valued array with imag as zero (0j).
        input_real_fp64 = scale*cp.random.random((shape[0], shape[1], shape[2])).astype(cp.float64)
        input_cmplx_interleave_fp64 = _pack_real_to_complex_interleave(input_real_fp64)

        return (input_real_fp64, input_cmplx_interleave_fp64)

    def _generate_data_cw(scale, dither=True):
        # Option 2: Generate CW
        # Store the input/output arrays as fp16 arrays twice as long, as complex32 is not yet available
        # Additonal note: CuPy is expecting the data as complex. To input real data create array as R+j0.

        # Real input with complex formatting. Generate N random samples and either use as 
        # a N-valued real array or create a complex-valued array with imag as zero (0j).
        
        f = shape[2]/32 # This should be 53.5MHz

        in_array = np.linspace(-(f*np.pi), f*np.pi, shape[2])
        if dither:
            input_real_fp64 = scale*np.cos(in_array).astype(np.float64) + 2**(-9)*np.random.random(shape[2]).astype(np.float64)
        else:
            input_real_fp64 = scale*np.cos(in_array).astype(np.float64)

        # temp_fft = np.fft.fft(input_real_fp64)
        # fft_power_spec = np.power(np.abs(temp_fft),2)

        # plt.figure(1)
        # plt.plot(10*np.log10(fft_power_spec))
        # plt.show()

        # plt.figure(2)
        # plt.plot(a_in)
        # plt.show()

        input_real_fp64 = input_real_fp64.reshape(shape)

        input_cmplx_interleave_fp64 = _pack_real_to_complex_interleave(input_real_fp64)

        return (input_real_fp64, input_cmplx_interleave_fp64)

    def _generate_constant(scale):
        # Option 3: Static vector array 
        d_first_real = scale*cp.ones((1,), dtype=np.float64)
        d_second_real = cp.zeros((shape[2]-1,), dtype=np.float64)
        input_real_fp64 = cp.concatenate([d_first_real, d_second_real])
        input_real_fp64 = input_real_fp64.reshape(input_real_fp64)
        input_cmplx_interleave_fp64 = _pack_real_to_complex_interleave(input_real_fp64)

        return (input_real_fp64, input_cmplx_interleave_fp64)

    if src == 'wgn':
        return _generate_data_wgn(scale)
    elif src == 'cw':
        return _generate_data_cw(scale, dither=True)
    elif src == 'const':
        return _generate_constant(scale)

def fft_cpu(input_cmplx_interleave_fp64):
    # ---- FFT FP64 with NumPy ----
    # Change interleave into complex128 format
    input_cmplx = input_cmplx_interleave_fp64.view(np.complex128)

    # Run FFT (FP64)(Complex128)
    fft_cpu_fp64 = np.fft.fftn(input_cmplx)

    # ---- FFT FP32 with NumPy ----
    # Run FFT (FP32)(Complex64)
    fft_cpu_fp32 = np.fft.fftn(input_cmplx.astype(np.complex64))

    # ---- FFT FP16 with NumPy ----
    # Run FFT (FP16)
    fft_cpu_fp16 = np.fft.fftn(input_cmplx_interleave_fp64.astype(np.float16))
    # Note: The input is not np.complex type (i.e. [r+j0]) but rather an array that is twice the length
    # where real and imag are separated as [r][j0]. Numpy (np.fft.fftn) seems to deal with the data the same
    # way. This is probably due to [r+j0] and [r][j0] being stored the same in the 'row-major' order (default).

    # This (above) yields the same results as first extracting the real-valued sequence and doing a np.fft.fft(real_input):
    # temp = np.real(input_cmplx_interleave_fp64)
    # temp = temp.astype(np.float16)
    # fft_cpu_fp16_real_input = np.fft.fft(temp)


    # Debug:
    # t1  = input_cmplx_interleave_fp64.astype(np.float16)
    # plt.figure(1)
    # plt.plot(input_cmplx_interleave_fp64[0][0] - t1[0][0])
    # plt.show()

    # fft_power_spec_fp64 = np.power(np.abs(fft_cpu_fp64[0][0][0:int(N/2)]),2)
    # fft_power_spec_fp32 = np.power(np.abs(fft_cpu_fp32[0][0][0:int(N/2)]),2)
    # fft_power_spec_fp16 = np.power(np.abs(fft_cpu_fp16[0][0][0:int(N/2)]),2)
    # plt.figure(1)
    # plt.plot(10*np.log10(fft_power_spec_fp16))
    # plt.plot(10*np.log10(fft_power_spec_fp64))
    # plt.figure(2)
    # plt.plot(10*np.log10(fft_power_spec_fp32))
    # plt.show()

    return (fft_cpu_fp64[0][0][0:int(N/2)], fft_cpu_fp32[0][0][0:int(N/2)], fft_cpu_fp16[0][0][0:int(N/2)])

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

def fft_fpga(filenames):
    fpga_fft = []
    fpga_cmplx = []
    for filename in filenames:
        for file in filename:
            with h5py.File(file, "r") as f:
                # List all groups
                print("Keys: %s" % f.keys())
                a_group_key = list(f.keys())[0]

                # Get the data
                fpga_fft.append(list(f[a_group_key]))

    for entry in fpga_fft:
        for i in range(len(entry[0])):
            if np.isnan(entry[0][i]) or (entry[0][i] == 0):
                entry[0][i] = 2**(-10)*np.random.random()

    fpga_cmplx.append(fpga_fft[0][0] +1j*fpga_fft[1][0])
    fpga_cmplx.append(fpga_fft[2][0] +1j*fpga_fft[3][0])

    # data_cmplx = data[0][0] + 1j*data[1][0]
    # fft_power_spec = np.square(np.abs(data_cmplx))

    # plt.figure(1)
    # plt.plot(data[0])
    # plt.show()

    # from os.path import dirname, join as pjoin
    # import scipy.io as sio
    # data_dir = pjoin(os.getcwd(), 'fft_analysis')
    # mat_fname = pjoin(data_dir, filename)
    # mat_contents = sio.loadmat(mat_fname)
    # a = 1
    # return (fpga_fft[0][0], fpga_fft[1][0], fpga_fft[2][0])
    return fpga_cmplx

def analyse_data(fft_cpu_out, fft_gpu_out, fpga_cmplx):
    fft_gpu_fp32_idx = 0
    fft_gpu_fp16_idx = 1
    fft_gpu_vk_idx = 2

    def _compute_mse(fft_cpu_out, fft_gpu_out, fpga_cmplx):
        # Compute MSE for CPU, GPU(FP32) and GPU(FP16)

        # CPU vs GPU
        # ----------        
        print('CPU vs GPU')
        print('----------')

        # CPU (FP64) vs GPU (FP32)(PyCUDA)
        cpu_fp64_gpu_fp32_diff = np.abs(fft_cpu_out[0] - fft_gpu_out[fft_gpu_fp32_idx])
        cpu_fp64_gpu_fp32_mse = np.sum(np.power(cpu_fp64_gpu_fp32_diff,2))/len(cpu_fp64_gpu_fp32_diff)
        print(f'CPU (FP64) vs GPU (FP32)(PyCUDA) MSE: {cpu_fp64_gpu_fp32_mse}')
        print(f'% of 8b LSB: {cpu_fp64_gpu_fp32_mse/2**(-7)*100}')
        print('')


        # CPU (FP64) vs GPU (FP32)(vkFFT)
        cpu_fp64_gpu_vkfp32_diff = np.abs(fft_cpu_out[0] - fft_gpu_out[fft_gpu_vk_idx])
        cpu_fp64_gpu_vkfp32_mse = np.sum(np.power(cpu_fp64_gpu_vkfp32_diff,2))/len(cpu_fp64_gpu_vkfp32_diff)
        print(f'CPU (FP64) vs GPU (FP32)(vkFFT) MSE: {cpu_fp64_gpu_vkfp32_mse}')
        print('Note: vkFFT version seems to flip channels around tone so MSE shows greater error.')
        print(f'% of 8b LSB: {cpu_fp64_gpu_vkfp32_mse/2**(-7)*100}')
        print('')

        # CPU (FP64) vs GPU (FP16)(CuPy)
        cpu_fp64_gpu_fp16_diff = np.abs(fft_cpu_out[0] - fft_gpu_out[fft_gpu_fp16_idx])
        cpu_fp64_gpu_fp16_mse = np.sum(np.power(cpu_fp64_gpu_fp16_diff,2))/len(cpu_fp64_gpu_fp16_diff)
        print(f'CPU (FP64) vs GPU (FP16)(CuPy) MSE: {cpu_fp64_gpu_fp16_mse}')
        print(f'% of 8b LSB: {cpu_fp64_gpu_fp16_mse/2**(-7)*100}')
        print('')

        # CPU (FP32) vs GPU (FP32)(PyCUDA)
        cpu_fp32_gpu_fp32_diff = np.abs(fft_cpu_out[1] - fft_gpu_out[fft_gpu_fp32_idx])
        cpu_fp32_gpu_fp32_mse = np.sum(np.power(cpu_fp32_gpu_fp32_diff,2))/len(cpu_fp32_gpu_fp32_diff)
        print(f'CPU (FP32) vs GPU (FP32)(PyCUDA) MSE: {cpu_fp32_gpu_fp32_mse}')
        print(f'% of 8b LSB: {cpu_fp32_gpu_fp32_mse/2**(-7)*100}')
        print('')

        # CPU (FP32) vs GPU (FP32)(vkFFT)
        cpu_fp32_gpu_vkfp32_diff = np.abs(fft_cpu_out[1] - fft_gpu_out[fft_gpu_vk_idx])
        cpu_fp32_gpu_vkfp32_mse = np.sum(np.power(cpu_fp32_gpu_vkfp32_diff,2))/len(cpu_fp32_gpu_vkfp32_diff)
        print(f'CPU (FP32) vs GPU (FP32)(vkFFT) MSE: {cpu_fp32_gpu_vkfp32_mse}')
        print('Note: vkFFT version seems to flip channels around tone so MSE shows greater error.')
        print(f'% of 8b LSB: {cpu_fp32_gpu_vkfp32_mse/2**(-7)*100}')
        print('')

        # CPU (FP32) vs GPU (FP16)(CuPy)
        cpu_fp32_gpu_fp16_diff = np.abs(fft_cpu_out[1] - fft_gpu_out[fft_gpu_fp16_idx])
        cpu_fp32_gpu_fp16_mse = np.sum(np.power(cpu_fp32_gpu_fp16_diff,2))/len(cpu_fp32_gpu_fp16_diff)
        print(f'CPU (FP32) vs GPU (FP16) MSE: {cpu_fp32_gpu_fp16_mse}')
        print(f'% of 8b LSB: {cpu_fp32_gpu_fp16_mse/2**(-7)*100}')
        print('')

        # CPU (FP16) vs GPU (FP32)(PyCUDA)
        cpu_fp16_gpu_fp32_diff = np.abs(fft_cpu_out[2] - fft_gpu_out[fft_gpu_fp32_idx])
        cpu_fp16_gpu_fp32_mse = np.sum(np.power(cpu_fp16_gpu_fp32_diff,2))/len(cpu_fp16_gpu_fp32_diff)
        print(f'CPU (FP16) vs GPU (FP32) MSE: {cpu_fp16_gpu_fp32_mse}')
        print(f'% of 8b LSB: {cpu_fp16_gpu_fp32_mse/2**(-7)*100}')
        print('')

        # CPU (FP16) vs GPU (FP16)(CuPy)
        cpu_fp16_gpu_fp16_diff = np.abs(fft_cpu_out[2] - fft_gpu_out[fft_gpu_fp16_idx])
        cpu_fp16_gpu_fp16_mse = np.sum(np.power(cpu_fp16_gpu_fp16_diff,2))/len(cpu_fp16_gpu_fp16_diff)
        print(f'CPU (FP16) vs GPU (FP16) MSE: {cpu_fp16_gpu_fp16_mse}')
        print(f'% of 8b LSB: {cpu_fp16_gpu_fp16_mse/2**(-7)*100}')
        print('')

        # GPU vs GPU
        # ----------
        print('GPU vs GPU')
        print('----------')

        # GPU (FP32)(PyCUDA) vs GPU (FP32)(vkFFT)
        gpu_fp32_gpu_vkfp32_diff = np.abs(fft_gpu_out[fft_gpu_fp32_idx] - np.conj(fft_gpu_out[fft_gpu_vk_idx]))
        gpu_fp32_gpu_vkfp32_mse = np.sum(np.square(gpu_fp32_gpu_vkfp32_diff))/len(gpu_fp32_gpu_vkfp32_diff)
        print(f'GPU (FP32)(PyCUDA) vs GPU (vkFFT)(FP32) MSE: {gpu_fp32_gpu_vkfp32_mse}')
        print(f'% of 8b LSB: {gpu_fp32_gpu_vkfp32_mse/2**(-7)*100}')
        print('')

        # GPU (FP32)(PyCUDA) vs GPU (FP16)(CuPy)
        gpu_fp32_gpu_fp16_diff = np.abs(fft_gpu_out[fft_gpu_fp32_idx] - fft_gpu_out[fft_gpu_fp16_idx])
        gpu_fp32_gpu_fp16_mse = np.sum(np.square(gpu_fp32_gpu_fp16_diff))/len(gpu_fp32_gpu_fp16_diff)
        print(f'GPU (FP32)(PyCUDA) vs GPU (FP16)(CuPy) MSE: {gpu_fp32_gpu_fp16_mse}')
        print(f'% of 8b LSB: {gpu_fp32_gpu_fp16_mse/2**(-7)*100}')
        print('')

        # plt.figure(1)
        # number_samples = len(fft_gpu_out[fft_gpu_vk_idx])*2
        # num_steps = 8
        # start_idx = 2038
        # end_idx = 2059
        # plt.plot(np.real(fft_gpu_out[fft_gpu_fp32_idx][start_idx:end_idx]))
        # plt.plot(np.real(fft_gpu_out[fft_gpu_vk_idx][start_idx:end_idx]))
        # plt.title(f'FFT: GPU (vkFFT Real - FP32) - {round(2048*1712e6/number_samples/1e6)}MHz')
        # labels = np.linspace(round(start_idx*1712e6/number_samples/1e6),round(end_idx*1712e6/number_samples/1e6), int(num_steps/2+1))
        # plt.xticks(np.arange(0, (end_idx-start_idx), step=5),labels=labels)
        # plt.xlabel('Frequency (MHz)')
        # plt.ylabel('Channel Magnitude')

        # plt.figure(2)
        # plt.plot(np.imag(fft_gpu_out[fft_gpu_fp32_idx][start_idx:end_idx]))
        # plt.plot(np.imag(fft_gpu_out[fft_gpu_vk_idx][start_idx:end_idx]))
        # plt.title(f'FFT: GPU (vkFFT Imag - FP32)- {round(2048*1712e6/number_samples/1e6)}MHz')
        # labels = np.linspace(round(start_idx*1712e6/number_samples/1e6),round(end_idx*1712e6/number_samples/1e6), int(num_steps/2+1))
        # plt.xticks(np.arange(0, (end_idx-start_idx), step=5),labels=labels)
        # plt.xlabel('Frequency (MHz)')
        # plt.ylabel('Channel Magnitude')

        # plt.figure(3)
        # plt.plot(np.imag(fft_gpu_out[fft_gpu_fp32_idx][start_idx:end_idx]))
        # plt.plot(np.imag(np.conj(fft_gpu_out[fft_gpu_vk_idx][start_idx:end_idx])))
        # plt.title(f'FFT: GPU (vkFFT Imag (Conj) - FP32) - {round(2048*1712e6/number_samples/1e6)}MHz')
        # labels = np.linspace(round(start_idx*1712e6/number_samples/1e6),round(end_idx*1712e6/number_samples/1e6), int(num_steps/2+1))
        # plt.xticks(np.arange(0, (end_idx-start_idx), step=5),labels=labels)
        # plt.xlabel('Frequency (MHz)')
        # plt.ylabel('Channel Magnitude')
        # plt.show()

        
        # CPU vs FPGA
        print('CPU vs FPGA')
        print('-----------')

        # CPU (FP32) vs FPGA (18b -> 8bit Quantized)
        cpu_fp32_fpga_nb_diff = np.abs(fft_cpu_out[1] - fpga_cmplx[0])
        cpu_fp32_fpga_nb_mse = np.sum(np.power(cpu_fp32_fpga_nb_diff,2))/len(cpu_fp32_fpga_nb_diff)
        print(f'CPU (FP32) vs FPGA (NB) MSE: {cpu_fp32_fpga_nb_mse}')
        print(f'% of 8b LSB: {cpu_fp32_fpga_nb_mse/2**(-7)*100}')
        print('')

        # CPU (FP32) vs FPGA (18b -> 8bit Quantized)
        cpu_fp32_fpga_wb_diff = np.abs(fft_cpu_out[1] - fpga_cmplx[1])
        cpu_fp32_fpga_wb_mse = np.sum(np.power(cpu_fp32_fpga_wb_diff,2))/len(cpu_fp32_fpga_wb_diff)
        print(f'CPU (FP32) vs FPGA (WB) MSE: {cpu_fp32_fpga_wb_mse}')
        print(f'% of 8b LSB: {cpu_fp32_fpga_wb_mse/2**(-7)*100}')
        print('') 

    def _compute_freq(all_ffts):
        measured_freq_and_fft_power_spec = []
        for fft_src in all_ffts:
            for fft in fft_src:
                fft_power_spec = np.power(np.abs(fft),2)
                fft_max = np.max(fft_power_spec)
                bin = np.where(fft_power_spec==fft_max)
                bin_freq_resolution = 1712e6/(len(fft_power_spec)*2)
                measured_freq_and_fft_power_spec.append((bin[0]*bin_freq_resolution, fft_power_spec))
        return measured_freq_and_fft_power_spec

    def _compute_sfdr(fft_power_spectrum):
        sfdr = []
        
        # Zero 'range' on either side of detected tone
        blank_range = 7000 #This is about 45MHz away from the fundamental

        for fft_entry in fft_power_spectrum:
            fft_power_spectrum = fft_entry[1].copy()

            # Compute fundamental bin
            fft_max_fundamental = np.max(fft_power_spectrum)
            fundamental_bin = np.where(fft_power_spectrum==fft_max_fundamental)
            fundamental_bin = fundamental_bin[0][0]

            if (fundamental_bin + blank_range) <= len(fft_power_spectrum):
                fft_power_spectrum[fundamental_bin:(fundamental_bin+blank_range)] = 0
            else:
                fft_power_spectrum[fundamental_bin:len(fft_power_spectrum)] = 0
            if (fundamental_bin - blank_range) >= 0:
                fft_power_spectrum[(fundamental_bin-blank_range):fundamental_bin] = 0
            else:
                fft_power_spectrum[0:fundamental_bin] = 0
            # Compute next dominant spike(tone).
            fft_max_second_tone = np.max(fft_power_spectrum)
            next_tone_bin = np.where(fft_power_spectrum==fft_max_second_tone)
            next_tone_bin = next_tone_bin[0][0]
            sfdr_dB = round(10*np.log10(fft_max_fundamental) - 10*np.log10(fft_max_second_tone),2)
            sfdr.append((sfdr_dB, fundamental_bin, next_tone_bin))

        return sfdr

    def display_sfdr(measured_freq_and_fft_power_spec, sfdr):
        num_steps = 8
        cpu_fp64_indx = 0
        cpu_fp32_indx = 1
        cpu_fp16_indx = 2
        gpu_fp32_indx = 3
        gpu_fp16_indx = 4
        gpu_vkfp32_indx = 5
        fpga_nb_indx = 6
        fpga_wb_indx = 7

        db_text_x_pos = 22e3
        db_text_y_pos = 60

        # CPU: FFT
        def disp_fft_cpu():
            # Numpy FFT FP64
            freq_cpu = measured_freq_and_fft_power_spec[cpu_fp64_indx][0]
            fft_power_spectrum_cpu = measured_freq_and_fft_power_spec[cpu_fp64_indx][1]
            number_samples = len(fft_power_spectrum_cpu)*2
            difference_dB_cpu_fp64 = sfdr[cpu_fp64_indx][0]

            # sfdr.append((freq_cpu, difference_dB)) # for printout
            fundamental_bin_cpu = sfdr[cpu_fp64_indx][1]
            next_tone_bin_cpu = sfdr[cpu_fp64_indx][2]

            plt.figure()
            markers_cpu = [fundamental_bin_cpu, next_tone_bin_cpu]
            print(f'difference_dB_cpu_fp64: {difference_dB_cpu_fp64}')
            plt.plot(10*np.log10(fft_power_spectrum_cpu), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin_cpu < len(fft_power_spectrum_cpu)/2:
                plt.text(db_text_x_pos, db_text_y_pos, f'SFDR ($\u25C6$): {difference_dB_cpu_fp64}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, db_text_y_pos, f'SFDR ($\u25C6$): {difference_dB_cpu_fp64}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: CPU (NumPy FP64)(Input Scale 0.1) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6)}MHz')
            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum_cpu)+len(fft_power_spectrum_cpu)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')


            # Numpy FFT FP32
            freq_cpu = measured_freq_and_fft_power_spec[cpu_fp32_indx][0]
            fft_power_spectrum_cpu = measured_freq_and_fft_power_spec[cpu_fp32_indx][1]
            number_samples = len(fft_power_spectrum_cpu)*2
            difference_dB_cpu_fp32 = sfdr[cpu_fp32_indx][0]

            # sfdr.append((freq_cpu, difference_dB)) # for printout
            fundamental_bin_cpu = sfdr[cpu_fp32_indx][1]
            next_tone_bin_cpu = sfdr[cpu_fp32_indx][2]

            plt.figure()
            markers_cpu = [fundamental_bin_cpu, next_tone_bin_cpu]
            print(f'difference_dB_cpu_fp32: {difference_dB_cpu_fp32}')
            plt.plot(10*np.log10(fft_power_spectrum_cpu), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin_cpu < len(fft_power_spectrum_cpu)/2:
                plt.text(db_text_x_pos, db_text_y_pos, f'SFDR ($\u25C6$): {difference_dB_cpu_fp32}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, db_text_y_pos, f'SFDR ($\u25C6$): {difference_dB_cpu_fp32}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: CPU (NumPy FP32)(Input Scale 0.1) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6)}MHz')
            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum_cpu)+len(fft_power_spectrum_cpu)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')

            # Numpy FFT FP16
            freq_cpu = measured_freq_and_fft_power_spec[cpu_fp16_indx][0]
            fft_power_spectrum_cpu = measured_freq_and_fft_power_spec[cpu_fp16_indx][1]
            number_samples = len(fft_power_spectrum_cpu)*2
            difference_dB_cpu_fp16 = sfdr[cpu_fp16_indx][0]

            # sfdr.append((freq_cpu, difference_dB)) # for printout
            fundamental_bin_cpu = sfdr[cpu_fp16_indx][1]
            next_tone_bin_cpu = sfdr[cpu_fp16_indx][2]

            plt.figure()
            markers_cpu = [fundamental_bin_cpu, next_tone_bin_cpu]
            print(f'difference_dB_cpu_fp16: {difference_dB_cpu_fp16}')
            plt.plot(10*np.log10(fft_power_spectrum_cpu), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin_cpu < len(fft_power_spectrum_cpu)/2:
                plt.text(db_text_x_pos, db_text_y_pos, f'SFDR ($\u25C6$): {difference_dB_cpu_fp16}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, db_text_y_pos, f'SFDR ($\u25C6$): {difference_dB_cpu_fp16}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: CPU (NumPy FP16)(Input Scale 0.1) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6)}MHz')
            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum_cpu)+len(fft_power_spectrum_cpu)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')

            plt.show()

        # GPU (FP32): FFT
        def disp_fft_gpu_fp32():
            freq = measured_freq_and_fft_power_spec[gpu_fp32_indx][0]
            fft_power_spectrum = measured_freq_and_fft_power_spec[gpu_fp32_indx][1]
            number_samples = len(fft_power_spectrum)*2
            difference_dB = sfdr[gpu_fp32_indx][0]

            # sfdr.append((freq_cpu, difference_dB)) # for printout
            fundamental_bin = sfdr[gpu_fp32_indx][1]
            next_tone_bin = sfdr[gpu_fp32_indx][2]

            plt.figure()
            markers_cpu = [fundamental_bin, next_tone_bin]
            print(f'difference_dB GPU PyCUDA FP32: {difference_dB}')
            plt.plot(10*np.log10(fft_power_spectrum), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin < len(fft_power_spectrum)/2:
                plt.text(db_text_x_pos, db_text_y_pos, f'SFDR ($\u25C6$): {difference_dB}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, db_text_y_pos, f'SFDR ($\u25C6$): {difference_dB}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: GPU (PyCUDA FP32)(Input Scale 0.1) - {round(fundamental_bin*1712e6/number_samples/1e6)}MHz')
            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum)+len(fft_power_spectrum)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')

        # GPU (FP16): FFT
        def disp_fft_gpu_fp16():
            freq = measured_freq_and_fft_power_spec[gpu_fp16_indx][0]
            fft_power_spectrum = measured_freq_and_fft_power_spec[gpu_fp16_indx][1]
            number_samples = len(fft_power_spectrum)*2
            difference_dB = sfdr[gpu_fp16_indx][0]

            # sfdr.append((freq_cpu, difference_dB)) # for printout
            fundamental_bin = sfdr[gpu_fp16_indx][1]
            next_tone_bin = sfdr[gpu_fp16_indx][2]

            plt.figure()
            markers_cpu = [fundamental_bin, next_tone_bin]
            print(f'difference_dB GPU CuPy FP16: {difference_dB}')
            plt.plot(10*np.log10(fft_power_spectrum), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin < len(fft_power_spectrum)/2:
                plt.text(db_text_x_pos, db_text_y_pos, f'SFDR Pol0 ($\u25C6$): {difference_dB}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, db_text_y_pos, f'SFDR Pol0: ($\u25C6$) {difference_dB}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: GPU (CuPy FP16)(Input Scale 0.1) - {round(fundamental_bin*1712e6/number_samples/1e6)}MHz')
            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum)+len(fft_power_spectrum)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')

        # GPU (FP32): vkFFT
        def disp_fft_gpu_vkfp32():
            freq = measured_freq_and_fft_power_spec[gpu_vkfp32_indx][0]
            fft_power_spectrum = measured_freq_and_fft_power_spec[gpu_vkfp32_indx][1]
            number_samples = len(fft_power_spectrum)*2
            difference_dB = sfdr[gpu_vkfp32_indx][0]

            # sfdr.append((freq_cpu, difference_dB)) # for printout
            fundamental_bin = sfdr[gpu_vkfp32_indx][1]
            next_tone_bin = sfdr[gpu_vkfp32_indx][2]

            plt.figure()
            markers_cpu = [fundamental_bin, next_tone_bin]
            print(f'difference_dB GPU(vkFFT FP32): {difference_dB}')
            plt.plot(10*np.log10(fft_power_spectrum), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin < len(fft_power_spectrum)/2:
                plt.text(db_text_x_pos, db_text_y_pos, f'SFDR Pol0 ($\u25C6$): {difference_dB}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, db_text_y_pos, f'SFDR Pol0: ($\u25C6$) {difference_dB}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: GPU (vkFFT FP32)(Input Scale 0.1) - {round(fundamental_bin*1712e6/number_samples/1e6)}MHz')
            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum)+len(fft_power_spectrum)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')
            plt.show()

        # FPGA:
        def disp_fpga():
            freq = measured_freq_and_fft_power_spec[fpga_nb_indx][0]
            fft_power_spectrum = measured_freq_and_fft_power_spec[fpga_nb_indx][1]
            number_samples = len(fft_power_spectrum)*2
            difference_dB = sfdr[fpga_nb_indx][0]

            # sfdr.append((freq_cpu, difference_dB)) # for printout
            fundamental_bin = sfdr[fpga_nb_indx][1]
            next_tone_bin = sfdr[fpga_nb_indx][2]

            plt.figure()
            markers_cpu = [fundamental_bin, next_tone_bin]
            print(f'difference_dB FPGA: {difference_dB}')
            plt.plot(10*np.log10(fft_power_spectrum), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin < len(fft_power_spectrum)/2:
                plt.text(0.25e4, -30, f'SFDR Pol0 ($\u25C6$): {difference_dB}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, -30, f'SFDR Pol0: ($\u25C6$) {difference_dB}dB', color='green', style='italic')
            # plt.title(f'SFDR FFT: FPGA (Xilinx) (8b Quant - Input Scale 0.1) - {round(fundamental_bin*214e6/number_samples/1e6)}MHz')
            plt.title(f'SFDR FFT: FPGA (Xilinx) (8b Quant - Input Scale 0.9) - {round(fundamental_bin*214e6/number_samples/1e6)}MHz')
            labels = np.linspace(0,(214e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum)+len(fft_power_spectrum)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')
           
           
            freq = measured_freq_and_fft_power_spec[fpga_wb_indx][0]
            fft_power_spectrum = measured_freq_and_fft_power_spec[fpga_wb_indx][1]
            number_samples = len(fft_power_spectrum)*2
            difference_dB = sfdr[fpga_wb_indx][0]

            # sfdr.append((freq_cpu, difference_dB)) # for printout
            fundamental_bin = sfdr[fpga_wb_indx][1]
            next_tone_bin = sfdr[fpga_wb_indx][2]
            plt.figure()
            markers_cpu = [fundamental_bin, next_tone_bin]
            print(f'difference_dB FPGA (CASPER): {difference_dB}')
            plt.plot(10*np.log10(fft_power_spectrum), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin < len(fft_power_spectrum)/2:
                plt.text(db_text_x_pos, -10, f'SFDR Pol0 ($\u25C6$): {difference_dB}dB', color='green', style='italic')
            else:
                plt.text(0.24e4, -10, f'SFDR Pol0: ($\u25C6$) {difference_dB}dB', color='green', style='italic')
            # plt.title(f'SFDR FFT: FPGA (CASPER) (8b Quant - Input Scale 0.1) - {round(fundamental_bin*1712e6/number_samples/1e6)}MHz')
            plt.title(f'SFDR FFT: FPGA (CASPER) (8b Quant - Input Scale 0.9) - {round(fundamental_bin*1712e6/number_samples/1e6)}MHz')
            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum)+len(fft_power_spectrum)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')
           
            plt.show()

        print('SFDR: CPU')
        print('---------')
        disp_fft_cpu()
        print('')

        print('SFDR: GPU')
        print('---------')
        disp_fft_gpu_fp32()
        disp_fft_gpu_fp16()
        disp_fft_gpu_vkfp32()
        print('')

        print('SFDR: FPGA')
        print('----------')
        disp_fpga()
        print('')

    _compute_mse(fft_cpu_out, fft_gpu_out, fpga_cmplx)
    measured_freq_and_fft_power_spec = _compute_freq([fft_cpu_out, fft_gpu_out, fpga_cmplx])
    sfdr = _compute_sfdr(measured_freq_and_fft_power_spec)
    
    # Display results
    display_sfdr(measured_freq_and_fft_power_spec, sfdr)

def main():
    # Generate data: Options, 'wgn', 'cw', 'const'
    input_real_fp64, input_cmplx_interleave_fp64 = generate_data('cw', scale=0.1)

    # Run GPU FFT's
    fft_gpu_out = fft_gpu(input_real_fp64, input_cmplx_interleave_fp64)

    # Run CPU(numpy) FFT
    fft_cpu_out = fft_cpu(input_cmplx_interleave_fp64)

    # Import Quantised 8bit (FPGA)
    # filenames_nb = ("fft_analysis/fft_real_scale_0_9_freq_53_5MHz_dither_2_15_shift_21930.h5", 
    #                 "fft_analysis/fft_imag_scale_0_9_freq_53_5MHz_dither_2_15_shift_21930.h5")

    # filenames_wb = ("fft_analysis/fft_q_real_scale_0_1_freq_53_5MHz_dither_2_11_shift_32766.h5", 
    #                 "fft_analysis/fft_q_imag_scale_0_1_freq_53_5MHz_dither_2_11_shift_32766.h5")

    filenames_nb = ("fft_real_scale_0_9_freq_53_5MHz_dither_2_15_shift_21930.h5", 
                    "fft_imag_scale_0_9_freq_53_5MHz_dither_2_15_shift_21930.h5")

    filenames_wb = ("fft_q_real_scale_0_9_freq_53_5MHz_dither_2_11_shift_65535.h5", 
                    "fft_q_imag_scale_0_9_freq_53_5MHz_dither_2_11_shift_65535.h5")

    fpga_cmplx = fft_fpga(filenames=(filenames_nb, filenames_wb))

    # Analyse results
    analyse_data(fft_cpu_out, fft_gpu_out, fpga_cmplx)

if __name__ == "__main__":
    main()