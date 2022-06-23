import config
import numpy as np
import cupy as cp

class CWGenerator():
    def __init__(self) -> None:
        self.shape = (1, 1, config.N)

    def _pack_real_to_complex_interleave(self, data_in):
        data_out = np.zeros((self.shape[0], self.shape[1], 2*self.shape[2]), dtype=np.float64)
        r = 0
        for n in range(self.shape[2]):
            data_out[0][0][r] = data_in[0][0][n]
            r +=2
        return data_out

    def _generate_data_wgn(self, wgn_scale):
        # Option 1: Generate WGN
        # Store the input/output arrays as fp16 arrays twice as long, as complex32 is not yet available
        # Additonal note: CuPy is expecting the data as complex. To input real data create array as R+j0.

        # Set seed for repeatability
        cp.random.seed(1)

        # Real input with complex formatting. Generate N random samples and either use as 
        # a N-valued real array or create a complex-valued array with imag as zero (0j).
        input_real_fp64 = wgn_scale*cp.random.normal((self.shape[0], self.shape[1], self.shape[2])).astype(cp.float64)
        input_cmplx_interleave_fp64 = CWGenerator._pack_real_to_complex_interleave(input_real_fp64)

        return (input_real_fp64, input_cmplx_interleave_fp64)

    def _generate_data_cw(self, cw_scale, cw_freq, wgn_scale, distribution):
        # Option 2: Generate CW
        # Store the input/output arrays as fp16 arrays twice as long, as complex32 is not yet available
        # Additonal note: CuPy is expecting the data as complex. To input real data create array as R+j0.

        # Real input with complex formatting. Generate N random samples and either use as 
        # a N-valued real array or create a complex-valued array with imag as zero (0j).
        
        bin_resolution = config.max_freq/(config.N/2)

        lin_range = cw_freq/bin_resolution

        in_array = np.linspace(-(lin_range*np.pi), lin_range*np.pi, self.shape[2])
        if wgn_scale > 0:
            if distribution == 'Gaussian':
                input_real_fp64 = cw_scale*np.cos(in_array).astype(np.float64) + wgn_scale*np.random.normal(size=(1,self.shape[2]))
            elif distribution == 'uniform':
                input_real_fp64 = cw_scale*np.cos(in_array).astype(np.float64) + np.random.uniform(-0.5, 0.5, size=(1,self.shape[2]))*wgn_scale
        else:
            input_real_fp64 = cw_scale*np.cos(in_array).astype(np.float64)

        # View locally to check if correct
        # temp_fft = np.fft.fft(input_real_fp64)
        # fft_power_spec = np.power(np.abs(temp_fft),2)

        # plt.figure(1)
        # plt.plot(10*np.log10(fft_power_spec))
        # plt.show()

        # plt.figure(2)
        # plt.plot(a_in)
        # plt.show()

        input_real_fp64 = input_real_fp64.reshape(self.shape)

        input_cmplx_interleave_fp64 = CWGenerator._pack_real_to_complex_interleave(self, input_real_fp64)

        return (input_real_fp64, input_cmplx_interleave_fp64)

    def _generate_constant(self, scale: float=0.99):
        # Option 3: Static vector array 
        d_first_real = scale*cp.ones((1,), dtype=np.float64)
        d_second_real = cp.zeros((self.shape[2]-1,), dtype=np.float64)
        input_real_fp64 = cp.concatenate([d_first_real, d_second_real])
        input_real_fp64 = input_real_fp64.reshape(input_real_fp64)
        input_cmplx_interleave_fp64 = CWGenerator._pack_real_to_complex_interleave(input_real_fp64)

        return (input_real_fp64, input_cmplx_interleave_fp64)

    def generate_data(self, signal: str ='cw', cw_scale: float=0.1, cw_freq: float=53.5e6, wgn_scale: float=2**(-9), distribution: str ='Gaussian'):
        if signal == 'wgn':
            return CWGenerator._generate_data_wgn(self, wgn_scale)
        elif signal == 'cw':
            return CWGenerator._generate_data_cw(self, cw_scale, cw_freq, wgn_scale, distribution)
        elif signal == 'const':
            return CWGenerator._generate_constant(self)