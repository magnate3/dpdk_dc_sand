import numpy as np
import config

class cw_analysis():

    def compute_measured_freq(samples):
        fft_result = np.fft.fft(samples)
        fft_result = fft_result[0:round(len(fft_result)/2)]
        fft_result = np.power(np.abs(fft_result),2)

        fft_max = np.max(fft_result)
        bin = np.where(fft_result==fft_max)
        bin_freq_resolution = 1712e6/len(samples)
        measured_freq = bin[0]*bin_freq_resolution
        return measured_freq, fft_result

    def compute_sfdr():
        pass

    async def run(samples):
        hist = []
        cw_min = []
        cw_max = []

        for pol in range(len(samples)):            
            hist.append(np.histogram(samples[pol],config.hist_res))
            cw_max.append(np.max(samples[pol]))
            cw_min.append(np.min(samples[pol]))

        return cw_max, cw_min, hist

    async def run_freq_checks(samples, freq):
        requested_vs_measured_freq = []
        for pol in range(len(samples)):
            # Compute frequency range
            measured_freq, fft_result = cw_analysis.compute_measured_freq(samples[pol])
            requested_vs_measured_freq.append((freq, measured_freq[0], fft_result))
        
            # Compute SFDR
            cw_analysis.compute_sfdr()

        return requested_vs_measured_freq