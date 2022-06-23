import numpy as np
import h5py

def fft_results(filenames):
    fpga_fft = []
    fpga_cmplx = []
    for filename in filenames:
        for file in filename:
            with h5py.File(file, "r") as f:
                # List all groups
                print("Keys: %s" % f.keys())
                a_group_key = list(f.keys())[0]

                # Get the data
                fpga_fft.append(list(f[a_group_key]))

    # any entry is nan or zero replace with a tiny amount of noise (1 lsb).
    for entry in fpga_fft:
        for i in range(len(entry[0])):
            if np.isnan(entry[0][i]) or (entry[0][i] == 0):
                entry[0][i] = 2**(-10)*np.random.normal()

    fpga_cmplx.append(fpga_fft[0][0] +1j*fpga_fft[1][0])
    fpga_cmplx.append(fpga_fft[2][0] +1j*fpga_fft[3][0])

    # data_cmplx = data[0][0] + 1j*data[1][0]
    # fft_power_spec = np.square(np.abs(data_cmplx))

    # plt.figure(1)
    # plt.plot(data[0])
    # plt.show()

    # from os.path import dirname, join as pjoin
    # import scipy.io as sio
    # data_dir = pjoin(os.getcwd(), 'fft_analysis')
    # mat_fname = pjoin(data_dir, filename)
    # mat_contents = sio.loadmat(mat_fname)
    # return (fpga_fft[0][0], fpga_fft[1][0], fpga_fft[2][0])
    return fpga_cmplx