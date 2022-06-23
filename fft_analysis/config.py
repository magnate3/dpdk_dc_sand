import os

CPLX = 2
BYTE_BITS = 8
SAMPLE_BITS = 10
N_POLS = 2

N = 2**16
max_freq = 856e6
shape = (1, 1, N)  # input array shape

# FPGA H5 files for import
dirname = os.path.dirname(__file__)
filenames_nb = (os.path.join(dirname, 'fpga_h5_files/fft_real_scale_0_9_freq_53_5MHz_dither_2_15_shift_21930.h5'), 
                os.path.join(dirname, 'fpga_h5_files/fft_imag_scale_0_9_freq_53_5MHz_dither_2_15_shift_21930.h5'))

filenames_wb = (os.path.join(dirname, 'fpga_h5_files/fft_q_real_scale_0_1_freq_53_5MHz_dither_2_11_shift_32766.h5'), 
                os.path.join(dirname, 'fpga_h5_files/fft_q_imag_scale_0_1_freq_53_5MHz_dither_2_11_shift_32766.h5'))