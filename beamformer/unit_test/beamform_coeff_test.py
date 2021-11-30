"""
Module for performing unit tests on the beamform operation using a Numba-based kernel.

The beamform operation occurs on a reordered block of data with the following dimensions:
    - uint16_t [n_batches][polarizations][n_channels][n_blocks][samples_per_block][n_ants][complexity]

    - Typical values for the dimensions
        - n_antennas (a) = 64
        - n_channels (c) = 128
        - n_samples_per_channel (t) = 256
        - polarisations (p) = 2, always
        - n_blocks = 16, always
        - samples_per_block calculated as n_samples_per_channel // n_blocks

Contains one test (parametrised):
    1. The first test uses the list of values present in test/test_parameters.py to run the
        kernel through a range of value combinations.
"""


import numpy as np
import pytest
from beamforming import matrix_multiply
from beamform_coeffs.beamformcoeff_kernel import beamform_coeff_kernel
from katsdpsigproc import accel
from unit_test import complex_mult_cpu, test_parameters
from unit_test.coeff_generator import CoeffGenerator
import time
import matplotlib.pyplot as plt


@pytest.mark.parametrize("batches", test_parameters.batches)
@pytest.mark.parametrize("n_ants", test_parameters.array_size)
@pytest.mark.parametrize("num_channels", test_parameters.num_channels)
@pytest.mark.parametrize("num_samples_per_channel", test_parameters.num_samples_per_channel)
@pytest.mark.parametrize("num_beams", test_parameters.num_beams)
@pytest.mark.parametrize("xeng_id", test_parameters.xeng_id)
def test_beamform_coeffs_parametrised(batches, n_ants, num_channels, num_samples_per_channel, num_beams, xeng_id):
    """
    Parametrised unit test of the beamform computation using Numba-based kernel.

    This unit test runs the computation on a combination of parameters indicated in test_parameters.py. The values
    parametrised are indicated in the parameter list, operating on a *single* batch. This unit test also invokes
    verification of the beamformed data.

    Parameters
    ----------
    batches: int
        Number of batches to process.
    n_ants: int
        The number of antennas from which data will be received.
    num_channels: int
        The number of frequency channels out of the FFT.
        NB: This is not the number of FFT channels per stream.
        The number of channels per stream is calculated from this value.
    num_samples_per_channel: int
        The number of time samples per frequency channel.

    This test:
        1. Populate a host-side array with random data in the range of the relevant dtype.
        2. Instantiate the beamformer complex multiplication and pass this input data to it.
        3. Grab the output, beamformed data.
        4. Verify it relative to the input array using a reference computed on the host CPU.
    """
    # 1. Array parameters
    n_channels_per_stream = num_channels // n_ants // 4
    samples_per_block = 16
    n_blocks = num_samples_per_channel // samples_per_block
    num_pols = 2
    coeff_gen = CoeffGenerator(batches, n_channels_per_stream, n_blocks, samples_per_block, n_ants, xeng_id)

    sample_period = 1/1712e6
    samples_delay = 5

    NumDelayVals = n_channels_per_stream * num_beams * n_ants

    gpu_phasor_real0 = []
    gpu_phasor_imag0 = []
    gpu_phasor_real1 = []
    gpu_phasor_imag1 = []

    cpu_phasor_real0 = []
    cpu_phasor_imag0 = []
    cpu_phasor_real1 = []
    cpu_phasor_imag1 = []

    for d in range(1):
        delay_vals = []

        # Make all the delays the same so the results should be identical per antenna-beam
        for i in range(NumDelayVals):
            delay_vals.append(np.single((samples_delay)*sample_period))
            delay_vals.append(np.single(0))
            delay_vals.append(np.single(np.pi/2+d*0.1))
            delay_vals.append(np.single(0))

        # or

        # Sweep the delay incrementally across all channels but keep ants(i.e. on top of each other) the same
        # for i in range(NumDelayVals):
        #     delay_vals.append(np.single((i/n_channels_per_stream)*samples_delay*sample_period))
        #     delay_vals.append(np.single(0))
        #     delay_vals.append(np.single(np.pi + d*0.1))
        #     delay_vals.append(np.single(0))

        # or

        # Sweep the delay incrementally across all channels and antennas
        # for i in range(NumDelayVals):
        #     delay_vals.append(np.single((i/(n_ants*n_channels_per_stream))*samples_delay*sample_period))
        #     delay_vals.append(np.single(0))
        #     delay_vals.append(np.single(np.pi + d*0.1))
        #     delay_vals.append(np.single(0))

            # delay_vals.append(np.single((i/NumDelayVals)*sample_period/3.0))
            # delay_vals.append(np.single(0))
            # delay_vals.append(np.single(1-(i/NumDelayVals)*sample_period/3.0))
            # delay_vals.append(np.single(0))

        # Change to numpy array and reshape
        delay_vals = np.array(delay_vals)
        delay_vals = delay_vals.reshape(n_channels_per_stream, num_beams, n_ants,4)

        # 3.2 Generate Coeffs on GPU
        GPU_coeff = beamform_coeff_kernel.coeff_gen(delay_vals, batches, num_pols, num_beams, n_channels_per_stream, num_channels, n_ants, xeng_id, sample_period)
        gpu_phasor_real0.append(GPU_coeff[0][0][1][2][0])
        gpu_phasor_imag0.append(GPU_coeff[0][0][1][2][1])

        gpu_phasor_real1.append(GPU_coeff[0][0][63][2][0])
        gpu_phasor_imag1.append(GPU_coeff[0][0][63][2][1])

        # 5. Run CPU version. This will be used to verify GPU reorder.
        CPU_coeff = coeff_gen.CPU_Coeffs(delay_vals, batches, num_pols, num_beams, n_channels_per_stream, num_channels, n_ants, xeng_id, sample_period)
        cpu_phasor_real0.append(GPU_coeff[0][0][1][2][0])
        cpu_phasor_imag0.append(GPU_coeff[0][0][1][2][1])

        cpu_phasor_real1.append(GPU_coeff[0][0][63][2][0])
        cpu_phasor_imag1.append(GPU_coeff[0][0][63][2][1])

    #
    # plt.plot(gpu_phasor_real0)
    # plt.plot(cpu_phasor_real0)
    # plt.show()

    # for m in range(3):
    #     for n in range(2):
    #         for o in range(64):
    #             for p in range(8):
    #                 for q in range(16):
    #                     if CPU_coeff[0][0][0][p][q] != GPU_coeff[0][0][0][p][q]:
    #                         print('Mismatch:', 'm', m, 'n:', n, 'o:', o, 'p:', p, 'q:', q)

    # 6. Verify the processed/returned result
    #    - Both the input and output data are ultimately of type np.int8
    np.testing.assert_array_equal(CPU_coeff, GPU_coeff)

    a = 1

if __name__ == "__main__":
    for a in range(len(test_parameters.array_size)):
        test_beamform_coeffs_parametrised(
            test_parameters.batches[0],
            test_parameters.array_size[0],
            test_parameters.num_channels[0],
            test_parameters.num_samples_per_channel[0],
            test_parameters.num_beams[0],
            test_parameters.xeng_id[0],
        )
