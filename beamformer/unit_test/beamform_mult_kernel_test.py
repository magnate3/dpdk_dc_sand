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
from beamform_coeffs.beamformcoeff_kernel import BeamformCoeffKernel
from beamforming import matrix_multiply
from katsdpsigproc import accel
from unit_test import complex_mult_cpu, test_parameters
from unit_test.coeff_generator import CoeffGenerator


@pytest.mark.parametrize("batches", test_parameters.batches)
@pytest.mark.parametrize("n_ants", test_parameters.array_size)
@pytest.mark.parametrize("n_channels", test_parameters.n_channels)
@pytest.mark.parametrize("n_samples_per_channel", test_parameters.n_samples_per_channel)
@pytest.mark.parametrize("n_beams", test_parameters.n_beams)
@pytest.mark.parametrize("xeng_id", test_parameters.xeng_id)
@pytest.mark.parametrize("samples_delay", test_parameters.samples_delay)
@pytest.mark.parametrize("phase", test_parameters.phase)
def test_beamform_parametrised(
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
    Parametrised unit test of the beamform computation using Numba-based kernel.

    This unit test runs the computation on a combination of parameters indicated in test_parameters.py. The values
    parametrised are indicated in the parameter list, operating on a *single* batch. This unit test also invokes
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
        1. Populate a host-side array with random data in the range of the relevant dtype.
        2. Instantiate the beamformer complex multiplication and pass this input data to it.
        3. Grab the output, beamformed data.
        4. Verify it relative to the input array using a reference computed on the host CPU.
    """
    # 1. Array parameters
    # NOTE: test_id is a temporary inclusion meant to identify which complex multiply to call.
    test_id = "kernel"

    n_channels_per_stream = n_channels // n_ants // 4
    samples_per_block = 16
    n_blocks = n_samples_per_channel // samples_per_block
    num_pols = 2

    sample_period = 1 / 1712e6
    num_delay_vals = n_channels_per_stream * n_beams * n_ants
    delay_vals = []

    for _ in range(num_delay_vals):
        delay_vals.append(np.single((samples_delay) * sample_period))
        delay_vals.append(np.single(0.0))
        delay_vals.append(np.single(phase))
        delay_vals.append(np.single(0.0))

    # Change to numpy array and reshape
    delay_vals = np.array(delay_vals)
    delay_vals = delay_vals.reshape(n_channels_per_stream, n_beams, n_ants, 4)

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda, interactive=False)
    queue = ctx.create_command_queue()

    # Create BeamformTemplate and link to buffer slots
    beamform_mult_template = matrix_multiply.MatrixMultiplyTemplate(
        ctx,
        n_ants=n_ants,
        n_channels=n_channels_per_stream,
        n_samples_per_channel=n_samples_per_channel,
        n_beams=n_beams,
        batches=batches,
        test_id=test_id,
    )

    beamform_mult = beamform_mult_template.instantiate(queue, test_id)
    beamform_mult.ensure_all_bound()

    bufcoeff_device = beamform_mult.buffer("inCoeffs")
    host_coeff = bufcoeff_device.empty_like()

    buf_samples_device = beamform_mult.buffer("inData")
    buf_samples_host = buf_samples_device.empty_like()

    buf_beamform_device = beamform_mult.buffer("outData")
    buf_beamform_host = buf_beamform_device.empty_like()

    # 3.1 Generate random input data
    # Inject random data for test.
    rng = np.random.default_rng(seed=2021)

    buf_samples_host[:] = rng.uniform(
        np.iinfo(buf_samples_host.dtype).min, np.iinfo(buf_samples_host.dtype).max, buf_samples_host.shape
    ).astype(buf_samples_host.dtype)

    # 3.2 Generate Coeffs
    gpu_coeff_gen = BeamformCoeffKernel(
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
    host_coeff[:] = gpu_coeff_gen.coeff_gen()

    # 4. Matrix Multiply: Transfer input sample array to GPU, run complex multiply kernel, transfer output array to CPU.
    bufcoeff_device.set(queue, host_coeff)
    buf_samples_device.set(queue, buf_samples_host)
    beamform_mult()
    buf_beamform_device.get(queue, buf_beamform_host)

    # 5. Run CPU version. This will be used to verify GPU reorder.
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
    cpu_coeffs = cpu_coeff_gen.cpu_coeffs()

    beamform_data_cpu = complex_mult_cpu.complex_mult(
        input_data=buf_samples_host,
        coeffs=cpu_coeffs,
        output_data_shape=buf_beamform_host.shape,
    )

    # 6. Verify the processed/returned result
    #    - Both the input and output data are ultimately of type np.int8
    np.testing.assert_allclose(beamform_data_cpu, buf_beamform_host, rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
    for _ in range(len(test_parameters.array_size)):
        test_beamform_parametrised(
            test_parameters.batches[0],
            test_parameters.array_size[0],
            test_parameters.n_channels[0],
            test_parameters.n_samples_per_channel[0],
            test_parameters.n_beams[0],
            test_parameters.xeng_id[0],
            test_parameters.samples_delay[0],
            test_parameters.phase[0],
        )
