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

import complex_mult_cpu
import numpy as np
import pytest
import test_parameters
from beamforming import matrix_multiply
from coeff_generator import CoeffGenerator
from katsdpsigproc import accel


@pytest.mark.parametrize("batches", test_parameters.batches)
@pytest.mark.parametrize("n_ants", test_parameters.array_size)
@pytest.mark.parametrize("num_channels", test_parameters.num_channels)
@pytest.mark.parametrize("num_samples_per_channel", test_parameters.num_samples_per_channel)
def test_beamform_parametrised(batches, n_ants, num_channels, num_samples_per_channel):
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
    # NOTE: test_id is a temporary inclusion meant to identify which complex multiply to call.
    test_id = "kernel"

    n_channels_per_stream = num_channels // n_ants // 4
    samples_per_block = 16
    n_blocks = num_samples_per_channel // samples_per_block
    coeff_gen = CoeffGenerator(batches, n_channels_per_stream, n_blocks, samples_per_block, n_ants)

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda, interactive=False)
    queue = ctx.create_command_queue()

    # Create BeamformTemplate and link to buffer slots
    beamform_mult_template = matrix_multiply.MatrixMultiplyTemplate(
        ctx,
        n_ants=n_ants,
        n_channels=n_channels_per_stream,
        n_samples_per_channel=num_samples_per_channel,
        batches=batches,
        test_id=test_id,
    )

    BeamformMult = beamform_mult_template.instantiate(queue, test_id)
    BeamformMult.ensure_all_bound()

    bufcoeff_device = BeamformMult.buffer("inCoeffs")
    host_coeff = bufcoeff_device.empty_like()

    bufSamples_device = BeamformMult.buffer("inData")
    bufSamples_host = bufSamples_device.empty_like()

    bufBeamform_device = BeamformMult.buffer("outData")
    bufBeamform_host = bufBeamform_device.empty_like()

    # 3.1 Generate random input data
    # Inject random data for test.
    rng = np.random.default_rng(seed=2021)

    bufSamples_host[:] = rng.uniform(
        np.iinfo(bufSamples_host.dtype).min, np.iinfo(bufSamples_host.dtype).max, bufSamples_host.shape
    ).astype(bufSamples_host.dtype)

    # 3.2 Generate Coeffs
    host_coeff[:] = coeff_gen.GPU_Coeffs_kernel()

    # 4. Matrix Multiply: Transfer input sample array to GPU, run complex multiply kernel, transfer output array to CPU.
    bufcoeff_device.set(queue, host_coeff)
    bufSamples_device.set(queue, bufSamples_host)
    BeamformMult()
    bufBeamform_device.get(queue, bufBeamform_host)

    # 5. Run CPU version. This will be used to verify GPU reorder.
    cpu_coeffs = coeff_gen.CPU_Coeffs()
    beamform_data_cpu = complex_mult_cpu.complex_mult(
        input_data=bufSamples_host,
        coeffs=cpu_coeffs,
        output_data_shape=bufBeamform_host.shape,
    )

    # 6. Verify the processed/returned result
    #    - Both the input and output data are ultimately of type np.int8
    np.testing.assert_array_equal(beamform_data_cpu, bufBeamform_host)


if __name__ == "__main__":
    for a in range(len(test_parameters.array_size)):
        test_beamform_parametrised(
            test_parameters.batches[0],
            test_parameters.array_size[a],
            test_parameters.num_channels[0],
            test_parameters.num_samples_per_channel[0],
        )
