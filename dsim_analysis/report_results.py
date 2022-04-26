# Display results from tests
import matplotlib.pyplot as plt
import numpy as np
import config

def display_wgn_results(results):
    for entry in results:
        print(f'Pol0 - Scale: {entry[1]}   Mean: {entry[2][0]}    StdDev: {entry[3][0]}    Var: {entry[4][0]}')
        print(f'Pol1 - Scale: {entry[1]}   Mean: {entry[2][1]}    StdDev: {entry[3][1]}    Var: {entry[4][1]}')

    # plot the 3rd noise level (0.25)
    plot_hist(results[2])
    plot_allan_var(results[2][5])

def display_cw_results(results):
    scale = []
    max_scale = []
    max_scale_pol0 = []
    max_scale_pol1 = []
    for entry in results:
        print(f'Pol0 - Scale: {entry[1]}    Max: {entry[2][0]}    Min: {entry[3][0]}')
        print(f'Pol1 - Scale: {entry[1]}    Max: {entry[2][1]}    Min: {entry[3][1]}')
        scale.append(entry[1])
        max_scale_pol0.append(entry[2][0])
        max_scale_pol1.append(entry[2][1])

    max_scale = (max_scale_pol0, max_scale_pol1)

    plot_hist(results[0])
    plot_linearity_scale(scale, max_scale)
    plot_linearity_difference(scale, max_scale)

def display_compare_measured_vs_requested_freq(results):
    for entry in results:
        print(f'Pol0 - Requested Frequency: {entry[0][0][0]}    Measured Frequency: {entry[0][0][1]}  Difference: {np.abs(entry[0][0][0] - entry[0][0][1])}')
        print(f'Pol1 - Requested Frequency: {entry[0][1][0]}    Measured Frequency: {entry[0][1][1]}  Difference: {np.abs(entry[0][1][0] - entry[0][1][1])}')
        # plot_channelised_data(entry[0])

def display_sfdr(results):
    sfdr = []
    num_chan = config.CHUNK_SAMPLES
    num_steps = 16

    for entry in results:
        freq = (entry[0][0][0])
        fft_power_spectrum_p0 = entry[0][0][2]
        fft_power_spectrum_p1 = entry[0][1][2]
        difference_dB_p0 = entry[1][0][0]
        difference_dB_p1 = entry[1][1][0]
        sfdr.append((freq, difference_dB_p0, difference_dB_p1)) # for printout
        fundamental_bin_p0 = entry[1][0][1]
        fundamental_bin_p1 = entry[1][1][1]
        next_tone_bin_p0 = entry[1][0][2]
        next_tone_bin_p1 = entry[1][1][2]

        plt.figure()
        markers_p0 = [fundamental_bin_p0, next_tone_bin_p0]
        markers_p1 = [fundamental_bin_p1, next_tone_bin_p1]

        print(difference_dB_p0)
        print(difference_dB_p1)
        print(fundamental_bin_p0)
        print(next_tone_bin_p0)
        plt.semilogy(fft_power_spectrum_p0, '-D', markevery=markers_p0, markerfacecolor='green', markersize=9)
        plt.semilogy(fft_power_spectrum_p1, '-D', markevery=markers_p1, markerfacecolor='purple', markersize=9)
        if fundamental_bin_p0 < len(fft_power_spectrum_p0)/2:
            plt.text(8.5e4, 1e9, f'SFDR Pol0 ($\u25C6$): {difference_dB_p0}dB', color='green', style='italic')
            plt.text(8.5e4, 1e8, f'SFDR Pol1 ($\u25C6$): {difference_dB_p1}dB', color='purple', style='italic')
        else:
            plt.text(0.25e4, 1e9, f'SFDR Pol0: ($\u25C6$) {difference_dB_p0}dB', color='green', style='italic')
            plt.text(0.25e4, 1e8, f'SFDR Pol1: ($\u25C6$) {difference_dB_p1}dB', color='purple', style='italic')
        plt.title(f'SFDR Pol0 and Pol1 - {round(fundamental_bin_p0*1712e6/config.CHUNK_SAMPLES/1e6)}MHz')
        labels = np.linspace(0,((1712e6/(config.CHUNK_SAMPLES)*num_chan)/2)/1e6, num_steps)
        labels = labels.round(0)
        plt.xticks(np.arange(0, num_chan, step=num_chan/num_steps),labels=labels)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('dB')
        plt.show()



    for entry in sfdr:
        print(f'Pol0 - Frequency: {entry[0]}   SFDR: {entry[1]}')
        print(f'Pol1 - Frequency: {entry[0]}   SFDR: {entry[2]}')

def display_freq_step(results):
    for entry in results[0]:
        # fft = entry[0]
        freq_step = entry[1]
        print(f'freq_step is: {freq_step}')

    plt.figure()
    num_chan = 32
    num_steps = 4
    markers_p0 = np.where(results[0][0][0] == np.max(results[0][0][0]))
    markers_p1 = np.where(results[0][1][0] == np.max(results[0][1][0]))
    plt.semilogy(results[0][0][0][0:num_chan], '-D', markevery=markers_p0[0], markerfacecolor='green', markersize=9)
    plt.semilogy(results[0][1][0][0:num_chan], '-D', markevery=markers_p1[0], markerfacecolor='green', markersize=9)
    plt.title(f'Channelised Data (Channels 0-{num_chan-1})')
    plt.text(5, 5e13, f'Pol0: ($\u25C6$) {round(results[0][0][1],2)}Hz', color='green', style='italic')
    plt.text(5, 1e13, f'Pol1: ($\u25C6$) {round(results[0][1][1],2)}Hz', color='purple', style='italic')
    labels = np.linspace(0,(1712e6/(config.CHUNK_SAMPLES)*num_chan), num_steps)
    labels = labels.round(0)
    plt.xticks(np.arange(0, num_chan, step=num_chan/num_steps),labels=labels)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('dB')
    plt.legend(['Pol0', 'Pol1'])
    plt.show()

def plot_channelised_data(channelised_data):
    plt.figure()
    plt.semilogy(channelised_data[0][2])
    plt.semilogy(channelised_data[1][2])
    plt.title('Channelised Data')
    plt.xlabel('FFT Bin')
    plt.ylabel('dB')
    plt.legend(['Pol0', 'Pol1'])
    plt.show()


def plot_hist(hist_results):
    plt.figure()
    bins = hist_results[0][0][1]
    plt.stairs(hist_results[0][0][0], hist_results[0][0][1], edgecolor="black", fill=True)
    plt.stairs(hist_results[0][1][0], hist_results[0][1][1], edgecolor="black", fill=True)
    plt.title('Histogram')
    # plt.xticks(np.linspace(-1, 1, int(np.round(config.hist_res/4))), np.linspace(0, len(hist_results[0][0][0]), config.hist_res))
    plt.xlabel('Bins')
    plt.ylabel('Count')
    plt.legend(['Pol0', 'Pol1'])
    plt.show()

def plot_allan_var(allan_var_results):
    t2_p0,ad_p0 = allan_var_results[0]
    t2_p1,ad_p1 = allan_var_results[1] 
    plt.figure()
    plt.plot(t2_p0,ad_p0)
    plt.plot(t2_p1,ad_p1)
    plt.title('Allan Variance')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('Time Cluster')
    plt.ylabel('Allan Deviation')
    plt.legend(['Pol0', 'Pol1'])
    plt.show()

def plot_linearity_scale(scale, max_scale):
    plt.figure()
    plt.semilogy(np.power(scale,2))
    plt.semilogy(np.power(max_scale[0],2))
    plt.semilogy(np.power(max_scale[1],2))
    plt.title('Linearty')
    plt.xlabel('Scaled Input Iteration')
    plt.ylabel('dB')
    plt.legend(['Reference', 'Pol0', 'Pol1'])
    plt.show()

def plot_linearity_difference(scale, max_scale):
    plt.figure()
    plt.plot(np.abs(np.array(scale) - np.array(max_scale[0])))
    plt.plot(np.abs(np.array(scale) - np.array(max_scale[0])))
    plt.title('Difference')
    plt.xlabel('Scaled Input Iteration')
    plt.ylabel('Magnitude of Difference')
    plt.legend(['Pol0', 'Pol1'])
    plt.show()
