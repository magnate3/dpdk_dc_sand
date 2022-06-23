import numpy as np
import matplotlib.pyplot as plt

def run(fft_cpu_out, fft_gpu_out, fpga_cmplx):
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
        # plt.legend(['PyCUDA FP32', 'vkFFT FP32'])
        # plt.xlabel('Frequency (MHz)')
        # plt.ylabel('Channel Magnitude')

        # plt.figure(2)
        # plt.plot(np.imag(fft_gpu_out[fft_gpu_fp32_idx][start_idx:end_idx]))
        # plt.plot(np.imag(fft_gpu_out[fft_gpu_vk_idx][start_idx:end_idx]))
        # plt.title(f'FFT: GPU (vkFFT Imag - FP32)- {round(2048*1712e6/number_samples/1e6)}MHz')
        # labels = np.linspace(round(start_idx*1712e6/number_samples/1e6),round(end_idx*1712e6/number_samples/1e6), int(num_steps/2+1))
        # plt.xticks(np.arange(0, (end_idx-start_idx), step=5),labels=labels)
        # plt.legend(['PyCUDA FP32', 'vkFFT FP32'])
        # plt.xlabel('Frequency (MHz)')
        # plt.ylabel('Channel Magnitude')

        # plt.figure(3)
        # plt.plot(np.imag(fft_gpu_out[fft_gpu_fp32_idx][start_idx:end_idx]))
        # plt.plot(np.imag(np.conj(fft_gpu_out[fft_gpu_vk_idx][start_idx:end_idx])))
        # plt.title(f'FFT: GPU (vkFFT Imag (Conj) - FP32) - {round(2048*1712e6/number_samples/1e6)}MHz')
        # labels = np.linspace(round(start_idx*1712e6/number_samples/1e6),round(end_idx*1712e6/number_samples/1e6), int(num_steps/2+1))
        # plt.xticks(np.arange(0, (end_idx-start_idx), step=5),labels=labels)
        # plt.legend(['PyCUDA FP32', 'vkFFT FP32 (Conj)'])
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
            # --------------
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
            plt.title(f'SFDR FFT: CPU (NumPy FP64)(Input Scale 0.1) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6,2)}MHz')
            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum_cpu)+len(fft_power_spectrum_cpu)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')


            # Numpy FFT FP32
            # --------------
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
            plt.title(f'SFDR FFT: CPU (NumPy FP32)(Input Scale 0.1) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6,2)}MHz')

            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum_cpu)+len(fft_power_spectrum_cpu)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')



            # SciPy FFT FP64 vs FP32
            # ----------------------
            fft_power_spectrum_cpu_fp64 = measured_freq_and_fft_power_spec[cpu_fp64_indx][1]
            fft_power_spectrum_cpu_fp32 = measured_freq_and_fft_power_spec[cpu_fp32_indx][1]
            number_samples = len(fft_power_spectrum_cpu_fp64)*2
            difference_dB_cpu_fp64 = sfdr[cpu_fp64_indx][0]

            fundamental_bin_cpu = sfdr[cpu_fp64_indx][1]
            next_tone_bin_cpu = sfdr[cpu_fp64_indx][2]

            plt.figure()
            plt.plot(10*np.log10(fft_power_spectrum_cpu_fp32), linestyle='--', label='FP32')
            plt.plot(10*np.log10(fft_power_spectrum_cpu_fp64), label='FP64')
            plt.legend()
            # plt.title(f'FFT: CPU (SciPy FP64)(Input Scale 0.1) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6,2)}MHz')
            plt.title(f'FFT: CPU (SciPy)(Input Scale 0.1) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6,2)}MHz')
            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum_cpu_fp64)+len(fft_power_spectrum_cpu_fp64)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')


            # SciPy FFT FP32
            # --------------
            # fft_power_spectrum_cpu_fp32 = measured_freq_and_fft_power_spec[cpu_fp32_indx][1]
            # number_samples = len(fft_power_spectrum_cpu_fp32)*2
            # difference_dB_cpu_fp32 = sfdr[cpu_fp32_indx][0]

            # # sfdr.append((freq_cpu, difference_dB)) # for printout
            # fundamental_bin_cpu = sfdr[cpu_fp32_indx][1]
            # next_tone_bin_cpu = sfdr[cpu_fp32_indx][2]

            # plt.figure()
            # plt.plot(10*np.log10(fft_power_spectrum_cpu_fp32))
            # plt.title(f'FFT: CPU (SciPy FP32)(Input Scale 0.1) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6,2)}MHz')
            # labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            # labels = labels.round(0)
            # plt.xticks(np.arange(0, (len(fft_power_spectrum_cpu_fp32)+len(fft_power_spectrum_cpu_fp32)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            # plt.xlabel('Frequency (MHz)')
            # plt.ylabel('dB')



            # Numpy FFT FP16
            # --------------
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
            plt.title(f'SFDR FFT: CPU (NumPy FP16)(Input Scale 0.1) - {round(fundamental_bin_cpu*1712e6/number_samples/1e6,2)}MHz')
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
            plt.title(f'SFDR FFT: GPU (PyCUDA FP32)(Input Scale 0.1) - {round(fundamental_bin*1712e6/number_samples/1e6,2)}MHz')
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
            plt.title(f'SFDR FFT: GPU (CuPy FP16)(Input Scale 0.1) - {round(fundamental_bin*1712e6/number_samples/1e6,2)}MHz')
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
            plt.title(f'SFDR FFT: GPU (vkFFT FP32)(Input Scale 0.1) - {round(fundamental_bin*1712e6/number_samples/1e6,2)}MHz')
            labels = np.linspace(0,(1712e6/2)/1e6, int(num_steps/2+1))
            labels = labels.round(0)
            plt.xticks(np.arange(0, (len(fft_power_spectrum)+len(fft_power_spectrum)/(number_samples/num_steps)), step=number_samples/num_steps),labels=labels)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('dB')
            plt.show()

        # FPGA:
        def disp_fpga():
            # Plot for Xilinx FFT FPGA results
            # --------------------------------
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
           
            # Plot for CASPER FFT FPGA results
            # --------------------------------
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