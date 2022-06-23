import numpy as np
import scipy
import scipy.fftpack
import config

N = config.N
shape = config.shape

def fft_cpu(input_cmplx_interleave_fp64):
    # ---- FFT FP64 with NumPy ----
    # Change interleave into complex128 format
    input_cmplx = input_cmplx_interleave_fp64.view(np.complex128)

    # Run FFT (FP64)(Complex128)
    # NumPy
    fft_cpu_fp64 = np.fft.fftn(input_cmplx)
    
    # SciPy
    fft_cpu_fp64 = scipy.fftpack.fftn(input_cmplx)

    # ---- FFT FP32 with NumPy ----
    # Run FFT (FP32)(Complex64)
    # NumPy
    fft_cpu_fp32 = np.fft.fftn(input_cmplx.astype(np.complex64))

    # SciPy
    fft_cpu_fp32 = scipy.fftpack.fftn(input_cmplx.astype(np.complex64))

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