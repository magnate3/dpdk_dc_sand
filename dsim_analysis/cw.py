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

    def compute_sfdr(fft_power_spectrum):
        # Compute fundamental bin
        fft_max_fundamental = np.max(fft_power_spectrum)
        fundamental_bin = np.where(fft_power_spectrum==fft_max_fundamental)
        fundamental_bin = fundamental_bin[0][0]

        # Zero 'range' on either side of detected tone
        blank_range = 2500
        if (fundamental_bin + blank_range) <= len(fft_power_spectrum):
            fft_power_spectrum[fundamental_bin:(fundamental_bin+blank_range)] = 0
        else:
            fft_power_spectrum[fundamental_bin:len(fft_power_spectrum)-1] = 0

        if (fundamental_bin - blank_range) >= 0:
            fft_power_spectrum[(fundamental_bin-blank_range):fundamental_bin] = 0
        else:
            fft_power_spectrum[0:fundamental_bin] = 0

        # Compute next dominant spike(tone).
        fft_max_second_tone = np.max(fft_power_spectrum)
        next_tone_bin = np.where(fft_power_spectrum==fft_max_second_tone)
        next_tone_bin = next_tone_bin[0][0]
        sfdr_dB = round(10*np.log10(fft_max_fundamental - fft_max_second_tone),2)

        return sfdr_dB, fundamental_bin, next_tone_bin


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
        sfdr = []
        for pol in range(len(samples)):
            # Compute frequency range
            measured_freq, fft_power_spectrum = cw_analysis.compute_measured_freq(samples[pol])
            requested_vs_measured_freq.append((freq, measured_freq[0], fft_power_spectrum))
        
            # Compute SFDR from precomputed FFT
            sfdr.append(cw_analysis.compute_sfdr(fft_power_spectrum.copy()))

        return requested_vs_measured_freq, sfdr