# Pinned memory based on example: https://gist.github.com/sjperkins/d9e6db1b2d6038febb72
# FFT (FP16) Example: https://docs.cupy.dev/en/stable/user_guide/fft.html
from cmath import log10, pi
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

import pycuda.autoinit
import pycuda.gpuarray as cua
from pyvkfft.fft import fftn
from pyvkfft.opencl import VkFFTApp
# from pyvkfft.cuda import VkFFTApp
from scipy.misc import ascent
import pyopencl as cl
import pyopencl.array as cla

N = 2**18
shape = (1, 1, N)  # input array shape

def generate_data(src, scale=0.5):
    
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
        f = shape[2]/32
        in_array = np.linspace(-(f*np.pi), f*np.pi, shape[2])
        if dither:
            input_real_fp64 = scale*np.cos(in_array).astype(np.float64) + 0.0001*np.random.random(shape[2]).astype(np.float64)
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

        # input_fp16 = input_cmplx_interleave_fp64[0].astype(np.float16)
        input_fp16 = input_real_fp64[0].astype(np.float16)
        input_fp32 = input_real_fp64[0].astype(np.float32)

        # stream = cuda.Stream()
        # d = cua.to_gpu(np.random.uniform(shape).astype(np.complex64))     
        # d1 = cua.empty_like(d)
        # app = VkFFTApp(d.shape, d.dtype, ndim=1, queue=cq, r2c=False, inplace=False)
        # cuda.Context.synchronize()
        # d1 = app.fft(d1,d)
        # cuda.Context.synchronize()

        # vkFFT FP16 input, FP32 out
        # np_data = np.random.uniform(0,1,(1,1024)).astype(np.float32)
        # np_fft = np.fft.fftn(np_data)

        # stream = cuda.Stream()
        # d0 = cua.to_gpu(np_data)
        # cuda.Context.synchronize()
        # app = VkFFTApp(np_data.shape, np_data.dtype, ndim=1, r2c=True, queue=stream, inplace=False)

        # dst_size = d0.size // d0.shape[-1] * (d0.shape[-1] // 2 + 1)
        # shape_new = (1,dst_size)
        # npzeros = np.zeros(shape_new)
        # vkout = cua.to_gpu(npzeros)

        # vkout = app.fft(d0,vkout)
        # cuda.Context.synchronize()
        # gpu_out_fp16 = vkout.get()

        # del app

        # vkFFT FP32 input, FP32 out
        input_fp32 = input_real_fp64[0].astype(np.float32)

        d = cla.to_device(cq,input_fp32)
        app = VkFFTApp(d.shape, d.dtype, ndim=1, r2c=True, queue=cq)
        vkfft_fp_32 = app.fft(d)

        # fft_out = d1.get()
        # fft_power_spec = np.power(np.abs(out[0]),2)

        # print('Spectrum')
        # plt.figure(1)
        # plt.plot(10*np.log10(fft_power _spec))
        # plt.show()
        # print('Spectrum')

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

def analyse_data(fft_cpu_out, fft_gpu_out):
    fft_gpu_fp32_idx = 0
    fft_gpu_fp16_idx = 1
    fft_gpu_vk_idx = 2


    def _compute_mse(fft_cpu_out, fft_gpu_out):
        # Compute MSE for CPU, GPU(FP32) and GPU(FP16)

        # GPU vs GPU
        print('')
        print('GPU vs GPU')
        print('----------')

        # GPU (FP32)(PyCUDA) vs GPU (FP16)(CuPy)
        gpu_fp32_gpu_fp16_diff = np.abs(fft_gpu_out[fft_gpu_fp32_idx] - fft_gpu_out[fft_gpu_fp16_idx])
        gpu_fp32_gpu_fp16_mse = np.sum(np.square(gpu_fp32_gpu_fp16_diff))/len(gpu_fp32_gpu_fp16_diff)
        print(f'GPU (FP32) vs GPU (FP16) MSE: {gpu_fp32_gpu_fp16_mse}')

        # GPU (FP32)(PyCUDA) vs GPU (FP32)(vkFFT)
        gpu_fp32_gpu_vkfp32_diff = np.abs(fft_gpu_out[fft_gpu_fp32_idx] - np.conj(fft_gpu_out[fft_gpu_vk_idx]))
        gpu_fp32_gpu_vkfp32_mse = np.sum(np.square(gpu_fp32_gpu_vkfp32_diff))/len(gpu_fp32_gpu_vkfp32_diff)
        print(f'GPU (FP32) vs GPU (vkFFT)(FP32) MSE: {gpu_fp32_gpu_vkfp32_mse}')
        print('')

        print(fft_gpu_out[fft_gpu_fp32_idx][8190])
        print(fft_gpu_out[fft_gpu_vk_idx][8190])
        print(np.conj(fft_gpu_out[fft_gpu_vk_idx][8190]))
        print('')
        print(fft_gpu_out[fft_gpu_fp32_idx][8191])
        print(fft_gpu_out[fft_gpu_vk_idx][8191])
        print(np.conj(fft_gpu_out[fft_gpu_vk_idx][8191]))
        print('')
        print(fft_gpu_out[fft_gpu_fp32_idx][8192])
        print(fft_gpu_out[fft_gpu_vk_idx][8192])
        print(np.conj(fft_gpu_out[fft_gpu_vk_idx][8192]))
        print('')
        print(fft_gpu_out[fft_gpu_fp32_idx][8193])
        print(fft_gpu_out[fft_gpu_vk_idx][8193])
        print(np.conj(fft_gpu_out[fft_gpu_vk_idx][8193]))
        print('')
        print(fft_gpu_out[fft_gpu_fp32_idx][8194])
        print(fft_gpu_out[fft_gpu_vk_idx][8194])
        print(np.conj(fft_gpu_out[fft_gpu_vk_idx][8194]))
        print('')

        plt.figure()

        plt.plot(np.imag(fft_gpu_out[fft_gpu_fp32_idx][8170:8210]))
        plt.plot(np.imag(np.conj(fft_gpu_out[fft_gpu_vk_idx][8170:8210])))

        plt.figure()
        plt.plot(np.real(fft_gpu_out[fft_gpu_fp32_idx][8170:8210]))
        plt.plot(np.real(np.conj(fft_gpu_out[fft_gpu_vk_idx][8170:8210])))

        # plt.figure()
        # plt.plot(fft_gpu_out[fft_gpu_vk_idx][10000:20000])
        # plt.title(f'Diff2')
        plt.show()

        # CPU vs GPU        
        print('CPU vs GPU')
        print('----------')

        # CPU (FP64) vs GPU (FP16)(CuPy)
        cpu_fp64_gpu_fp16_diff = np.abs(fft_cpu_out[0] - fft_gpu_out[fft_gpu_fp16_idx])
        cpu_fp64_gpu_fp16_mse = np.sum(np.power(cpu_fp64_gpu_fp16_diff,2))/len(cpu_fp64_gpu_fp16_diff)
        print(f'CPU (FP64) vs GPU (FP16) MSE: {cpu_fp64_gpu_fp16_mse}')

        # CPU (FP32) vs GPU (FP16)(CuPy)
        cpu_fp32_gpu_fp16_diff = np.abs(fft_cpu_out[1] - fft_gpu_out[fft_gpu_fp16_idx])
        cpu_fp32_gpu_fp16_mse = np.sum(np.power(cpu_fp32_gpu_fp16_diff,2))/len(cpu_fp32_gpu_fp16_diff)
        print(f'CPU (FP32) vs GPU (FP16) MSE: {cpu_fp32_gpu_fp16_mse}')

        # CPU (FP16) vs GPU (FP16)(CuPy)
        cpu_fp16_gpu_fp16_diff = np.abs(fft_cpu_out[2] - fft_gpu_out[fft_gpu_fp16_idx])
        cpu_fp16_gpu_fp16_mse = np.sum(np.power(cpu_fp16_gpu_fp16_diff,2))/len(cpu_fp16_gpu_fp16_diff)
        print(f'CPU (FP16) vs GPU (FP16) MSE: {cpu_fp16_gpu_fp16_mse}')

        # CPU (FP64) vs GPU (FP32)(PyCUDA)
        cpu_fp64_gpu_fp32_diff = np.abs(fft_cpu_out[0] - fft_gpu_out[fft_gpu_fp32_idx])
        cpu_fp64_gpu_fp32_mse = np.sum(np.power(cpu_fp64_gpu_fp32_diff,2))/len(cpu_fp64_gpu_fp32_diff)
        print(f'CPU (FP64) vs GPU (FP32) MSE: {cpu_fp64_gpu_fp32_mse}')

        # CPU (FP32) vs GPU (FP32)(PyCUDA)
        cpu_fp32_gpu_fp32_diff = np.abs(fft_cpu_out[1] - fft_gpu_out[fft_gpu_fp32_idx])
        cpu_fp32_gpu_fp32_mse = np.sum(np.power(cpu_fp32_gpu_fp32_diff,2))/len(cpu_fp32_gpu_fp32_diff)
        print(f'CPU (FP32) vs GPU (FP32) MSE: {cpu_fp32_gpu_fp32_mse}')
        
        # CPU (FP16) vs GPU (FP32)(PyCUDA)
        gpu_fp32_gpu_fp16_diff = np.abs(fft_cpu_out[2] - fft_gpu_out[fft_gpu_fp32_idx])
        gpu_fp32_gpu_fp16_mse = np.sum(np.power(gpu_fp32_gpu_fp16_diff,2))/len(gpu_fp32_gpu_fp16_diff)
        print(f'GPU (FP32) vs GPU (FP16) MSE: {gpu_fp32_gpu_fp16_mse}')
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
        for fft_entry in fft_power_spectrum:
            fft_power_spectrum = fft_entry[1].copy()

            # Compute fundamental bin
            fft_max_fundamental = np.max(fft_power_spectrum)
            fundamental_bin = np.where(fft_power_spectrum==fft_max_fundamental)
            fundamental_bin = fundamental_bin[0][0]
            # Zero 'range' on either side of detected tone
            blank_range = 15000 #This is about 98MHz away from the fundamental
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
                plt.text(8.5e4, 70, f'SFDR ($\u25C6$): {difference_dB_cpu_fp64}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, 70, f'SFDR ($\u25C6$): {difference_dB_cpu_fp64}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: CPU (FP64) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6)}MHz')
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
                plt.text(8.5e4, 70, f'SFDR ($\u25C6$): {difference_dB_cpu_fp32}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, 70, f'SFDR ($\u25C6$): {difference_dB_cpu_fp32}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: CPU (FP32) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6)}MHz')
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
                plt.text(8.5e4, 70, f'SFDR ($\u25C6$): {difference_dB_cpu_fp16}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, 70, f'SFDR ($\u25C6$): {difference_dB_cpu_fp16}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: CPU (FP16) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6)}MHz')
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
            print(f'difference_dB GPU FP32: {difference_dB}')
            plt.plot(10*np.log10(fft_power_spectrum), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin < len(fft_power_spectrum)/2:
                plt.text(8.5e4, 70, f'SFDR ($\u25C6$): {difference_dB}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, 70, f'SFDR ($\u25C6$): {difference_dB}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: GPU(FP32) - {round(fundamental_bin*1712e6/number_samples/1e6)}MHz')
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
            print(f'difference_dB GPU FP16: {difference_dB}')
            plt.plot(10*np.log10(fft_power_spectrum), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin < len(fft_power_spectrum)/2:
                plt.text(8.5e4, 70, f'SFDR Pol0 ($\u25C6$): {difference_dB}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, 70, f'SFDR Pol0: ($\u25C6$) {difference_dB}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: GPU(FP16) - {round(fundamental_bin*1712e6/number_samples/1e6)}MHz')
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
            print(f'difference_dB GPU FP32(vkFFT): {difference_dB}')
            plt.plot(10*np.log10(fft_power_spectrum), '-D', markevery=markers_cpu, markerfacecolor='green', markersize=9)

            if fundamental_bin < len(fft_power_spectrum)/2:
                plt.text(8.5e4, 70, f'SFDR Pol0 ($\u25C6$): {difference_dB}dB', color='green', style='italic')
            else:
                plt.text(0.25e4, 70, f'SFDR Pol0: ($\u25C6$) {difference_dB}dB', color='green', style='italic')
            plt.title(f'SFDR FFT: GPU FP32(vkFFT) - {round(fundamental_bin*1712e6/number_samples/1e6)}MHz')
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

    _compute_mse(fft_cpu_out, fft_gpu_out)
    measured_freq_and_fft_power_spec = _compute_freq([fft_cpu_out, fft_gpu_out])
    sfdr = _compute_sfdr(measured_freq_and_fft_power_spec)
    
    # Display results
    display_sfdr(measured_freq_and_fft_power_spec, sfdr)

def fft_fpga(filename):
    import h5py
    # filename = "fft_analysis/fft_re.h5"
    filename = "fft_re.h5"

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
        # extract = data

        # print(data[0:5])

        plt.figure(1)
        plt.plot(data[0])
        plt.show()


    # from os.path import dirname, join as pjoin
    # import scipy.io as sio
    # data_dir = pjoin(os.getcwd(), 'fft_analysis')
    # mat_fname = pjoin(data_dir, filename)
    # mat_contents = sio.loadmat(mat_fname)
    # a = 1

def main():
    # Generate data: Options, 'wgn', 'cw', 'const'
    input_real_fp64, input_cmplx_interleave_fp64 = generate_data('cw', scale=0.1)

    # Run GPU FFT's
    fft_gpu_out = fft_gpu(input_real_fp64, input_cmplx_interleave_fp64)

    # Run CPU(numpy) FFT
    fft_cpu_out = fft_cpu(input_cmplx_interleave_fp64)

    # Import Quantised 8bit (FPGA)
    # fft_fpga_8bit = fft_fpga(filename='fft_re.hdf5')

    # Analyse results
    analyse_data(fft_cpu_out, fft_gpu_out)

if __name__ == "__main__":
    main()