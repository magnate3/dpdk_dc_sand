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

    def compute_beat_freq(sample_set):
        fft_set = []
        if len(sample_set) >= 2:
            for i in range(len(sample_set)-1):
                beat_signal = sample_set[i]*sample_set[i+1]
                fft_beat = np.fft.fft(beat_signal)
                fft_beat = fft_beat[0:1024]
                fft_beat = np.power(np.abs(fft_beat),2)
                fft_max = np.max(fft_beat)
                bin = np.where(fft_beat==fft_max)
                bin_freq_resolution = 1712e6/len(beat_signal)
                measured_freq = bin[0][0]*bin_freq_resolution
                print(measured_freq)
                fft_set.append((fft_beat, measured_freq))
        return fft_set

            

    def compute_sfdr(fft_power_spectrum):
        # Compute fundamental bin
        fft_max_fundamental = np.max(fft_power_spectrum)
        fundamental_bin = np.where(fft_power_spectrum==fft_max_fundamental)
        fundamental_bin = fundamental_bin[0][0]

        # Zero 'range' on either side of detected tone
        blank_range = 15000
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
    
    async def run_freq_step(sample_set):
        sample_sets_pol0 = []
        sample_sets_pol1 = []

        for sample in sample_set:
            print(sample[0][0])
            sample_sets_pol0.append(sample[0])
            sample_sets_pol1.append(sample[1])

        beat_freq_pol0 = cw_analysis.compute_beat_freq(sample_sets_pol0)
        beat_freq_pol1 = cw_analysis.compute_beat_freq(sample_sets_pol1)

        return (beat_freq_pol0, beat_freq_pol1)