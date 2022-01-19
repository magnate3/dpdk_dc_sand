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
from beamforming.beamform_coeffs import CoeffGeneratorTemplate
from katsdpsigproc import accel

# from beamform_coeffs.beamformcoeff_kernel import BeamformCoeffKernel
from unit_test import test_parameters
from unit_test.coeff_generator import CoeffGenerator


@pytest.mark.parametrize("batches", test_parameters.batches)
@pytest.mark.parametrize("n_ants", test_parameters.array_size)
@pytest.mark.parametrize("n_channels", test_parameters.n_channels)
@pytest.mark.parametrize("n_samples_per_channel", test_parameters.n_samples_per_channel)
@pytest.mark.parametrize("n_beams", test_parameters.n_beams)
@pytest.mark.parametrize("xeng_id", test_parameters.xeng_id)
@pytest.mark.parametrize("samples_delay", test_parameters.samples_delay)
@pytest.mark.parametrize("phase", test_parameters.phase)
def test_beamform_coeffs(
    batches: int,
    n_ants: int,
    n_channels: int,
    n_samples_per_channel: int,
    n_beams: int,
    xeng_id: int,
    samples_delay: int,
    phase: float,
):
    """
    Parametrised unit test of the beamform coefficient generation using Numba-based kernel.

    This unit test runs the generation on a combination of parameters indicated in test_parameters.py. The values
    parametrised are indicated in the parameter list, operating on a multiple batches. This unit test also invokes
    verification of the beamformed data.

    Parameters
    ----------
    batches:
        Number of batches to process.
    n_ants:
        The number of antennas from which data will be received.
    n_channels:
        The number of frequency channels out of the FFT.
        NB: This is not the number of FFT channels per stream.
        The number of channels per stream is calculated from this value.
    n_samples_per_channel:
        The number of time samples per frequency channel.
    n_beams:
        The number of beams that will be steered.
    xeng_id:
        Identify of the XEngine. This is used to compute the actual channel numbers processed per engine.
    samples_delay:
        Delay in ADC samples that should be applied.
    phase:
        Phase value in radians to be applied.

    This test:
        1. Populate a host-side array with sample delay and phase data.
        2. Instantiate the beamformer coefficient generator and pass delay values as input data to it.
        3. Grab the output, beamformed coefficients per antenna-beam.
        4. Verify it relative to the input array using a reference computed on the host CPU.
    """
    # 1. Array parameters
    n_channels_per_stream = n_channels // n_ants // 4
    samples_per_block = 16
    n_blocks = n_samples_per_channel // samples_per_block
    num_pols = 2

    sample_period = 1 / 1712e6

    num_delay_vals = n_channels_per_stream * n_beams * n_ants

    delay_vals = []

    # 2. Make all the delays the same so the results should be identical per antenna-beam
    for _ in range(num_delay_vals):
        delay_vals.append(np.single((samples_delay) * sample_period))
        delay_vals.append(np.single(0))
        delay_vals.append(np.single(phase))
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

    # Change to numpy array and reshape
    delay_vals = np.array(delay_vals)
    delay_vals = delay_vals.reshape(n_channels_per_stream, n_beams, n_ants, 4)

    cpu_coeff_gen = CoeffGenerator(
        delay_vals,
        batches,
        num_pols,
        n_channels_per_stream,
        n_channels,
        n_blocks,
        samples_per_block,
        n_ants,
        n_beams,
        xeng_id,
        sample_period,
    )
    # gpu_coeff_gen = BeamformCoeffKernel(
    #     delay_vals,
    #     batches,
    #     num_pols,
    #     n_channels_per_stream,
    #     n_channels,
    #     n_blocks,
    #     samples_per_block,
    #     n_ants,
    #     n_beams,
    #     xeng_id,
    #     sample_period,
    # )

    # Create context and command queue
    ctx = accel.create_some_context(device_filter=lambda device: device.is_cuda)
    queue = ctx.create_command_queue()

    # 3. Generate Coeffs on GPU
    coeff_template = CoeffGeneratorTemplate(
        ctx,
        batches,
        num_pols,
        n_channels_per_stream,
        n_channels,
        n_blocks,
        samples_per_block,
        n_ants,
        n_beams,
        xeng_id,
        sample_period,
    )

    beamform_coeffs = coeff_template.instantiate(queue)
    beamform_coeffs.ensure_all_bound()

    # Create host buffers
    buf_delay_vals_device = beamform_coeffs.buffer("delay_vals")
    host_delay_vals = buf_delay_vals_device.empty_like()

    bufcoeff_device = beamform_coeffs.buffer("outCoeffs")
    host_gpu_coeff = bufcoeff_device.empty_like()

    # Copy delay_vals from host to device
    host_delay_vals = delay_vals
    buf_delay_vals_device.set(queue, host_delay_vals)

    # Run gpu coeff kernel
    beamform_coeffs()

    # Get beamforming coeffs from gpu memory
    bufcoeff_device.get(queue, host_gpu_coeff)

    # 4. Run CPU version. This will be used to verify GPU reorder.
    cpu_coeff = cpu_coeff_gen.cpu_coeffs()

    # 5. Verify the processed/returned result
    #    - Both the input and output data are ultimately of type np.int8
    np.testing.assert_array_equal(cpu_coeff, host_gpu_coeff)


if __name__ == "__main__":
    for _ in range(len(test_parameters.array_size)):
        test_beamform_coeffs(
            test_parameters.batches[0],
            test_parameters.array_size[0],
            test_parameters.n_channels[0],
            test_parameters.n_samples_per_channel[0],
            test_parameters.n_beams[0],
            test_parameters.xeng_id[0],
            test_parameters.samples_delay[0],
            test_parameters.phase[0],
        )
