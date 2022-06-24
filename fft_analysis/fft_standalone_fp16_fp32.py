from distutils.command.config import config
import cw_generator
import fft_fpga
import fft_gpu
import fft_cpu
import analyse_data
import config

def main():
    # Generate data: Options, 'wgn', 'cw', 'const'
    cw_scale=0.9
    cw_freq=53.5e6
    cw = cw_generator.CWGenerator()
    input_real_fp64, input_cmplx_interleave_fp64 = cw.generate_data(signal='cw', cw_scale=cw_scale, cw_freq=cw_freq, wgn_scale=2**(-10), distribution='Gaussian')

    # Run GPU FFT's
    fft_gpu_out = fft_gpu.fft_gpu(input_real_fp64, input_cmplx_interleave_fp64)

    # Run CPU(numpy) FFT
    fft_cpu_out = fft_cpu.fft_cpu(input_cmplx_interleave_fp64)

    # Import Quantised 8bit (FPGA)
    fpga_cmplx = fft_fpga.fft_results(filenames=(config.filenames_nb, config.filenames_wb))

    # Analyse results
    analyse_data.run(fft_cpu_out, fft_gpu_out, fpga_cmplx, cw_scale=cw_scale, cw_freq=cw_freq)

if __name__ == "__main__":
    main()