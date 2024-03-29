"""
Module for performing unit tests on the complete beamform operation using a Numba-based kernel.

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
from beamforming import reorder as cpu_reorder
from beamforming.beamform_op_sequence import OpSequenceTemplate
from katsdpsigproc import accel
from unit_test import complex_mult_cpu, test_parameters
from unit_test.coeff_generator_cpu import CoeffGenerator


@pytest.mark.parametrize("n_batches", test_parameters.n_batches)
@pytest.mark.parametrize("n_ants", test_parameters.n_ants)
@pytest.mark.parametrize("n_channels", test_parameters.n_channels)
@pytest.mark.parametrize("n_samples_per_channel", test_parameters.n_samples_per_channel)
@pytest.mark.parametrize("n_beams", test_parameters.n_beams)
@pytest.mark.parametrize("xeng_id", test_parameters.xeng_id)
@pytest.mark.parametrize("samples_delay", test_parameters.samples_delay)
@pytest.mark.parametrize("phase", test_parameters.phase)
def test_beamform_op_sequence(
    n_batches: int,
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
    n_batches:
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

    This test will:
        1. Populate a host-side array with random data in the range of the relevant dtype.
        2. Instantiate the pre-beamform reorder and pass data to it.
        3. Instantiate the beamform coefficient generator.
        4. Instantiate the beamformer complex multiplication.
        5. Grab the output, beamformed data.
        6. Verify it relative to the input array using a reference computed on the host CPU.
    """
    # 1. Array parameters

    n_channels_per_stream = n_channels // n_ants // 4
    samples_per_block = 16
    n_blocks = n_samples_per_channel // samples_per_block
    n_pols = 2

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
    ctx = accel.create_some_context(
        device_filter=lambda x: x.is_cuda, interactive=False
    )
    queue = ctx.create_command_queue()

    # Create compound
    op_template = OpSequenceTemplate(
        ctx,
        n_batches,
        n_pols,
        n_channels_per_stream,
        n_channels,
        n_blocks,
        samples_per_block,
        n_ants,
        n_beams,
        xeng_id,
        sample_period,
        n_samples_per_channel,
    )

    # Instantiate operational sequence
    op = op_template.instantiate(queue)
    op.ensure_all_bound()

    # Create host buffers
    buf_delay_vals_device = op.beamform_coeff.buffer("delay_vals")
    host_delay_vals = buf_delay_vals_device.empty_like()
    host_delay_vals[:] = delay_vals

    buf_data_in_device = op.prebeamform_reorder.buffer("inSamples")
    host_data_in = buf_data_in_device.empty_like()

    buf_beamform_data_out_device = op.beamform_mult.buffer("outData")
    host_beamform_data_out = buf_beamform_data_out_device.empty_like()

    # 3.1 Generate random input data
    # Inject random data for test.
    rng = np.random.default_rng(seed=2021)

    host_data_in[:] = rng.uniform(
        np.iinfo(host_data_in.dtype).min,
        np.iinfo(host_data_in.dtype).max,
        host_data_in.shape,
    ).astype(host_data_in.dtype)

    # 4. Beamforming (Coefficient generation and Matrix Multiply):
    # 4.1 Transfer input sample array to GPU;
    # 4.2 Transfer delay_vals for coefficient computation;
    # 4.3 Run pre-beamform reorder, coefficient generation, and complex multiply kernel;
    # 4.3 Transfer output array to CPU.
    buf_data_in_device.set(queue, host_data_in)
    buf_delay_vals_device.set(queue, host_delay_vals)

    # Run the operational sequence
    op()

    # Grab the beamformed data
    buf_beamform_data_out_device.get(queue, host_beamform_data_out)

    # 5. Run CPU coeff generator, reorder and beamformer
    # Run CPU coeff generator.
    cpu_coeff_gen = CoeffGenerator(
        delay_vals,
        n_batches,
        n_pols,
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

    # Run CPU reorder.
    reorder_output_data_cpu = cpu_reorder.reorder(
        input_data=host_data_in,
        input_data_shape=host_data_in.shape,
        output_data_shape=op.beamform_mult.buffer("inData").shape,
    )

    # Run CPU beamformer.
    beamform_data_cpu = complex_mult_cpu.complex_mult(
        input_data=reorder_output_data_cpu,
        coeffs=cpu_coeffs,
        output_data_shape=host_beamform_data_out.shape,
    )

    # 6. Verify the processed/returned result
    #    - Both the input and output data are ultimately of type np.int8
    np.testing.assert_allclose(
        beamform_data_cpu, host_beamform_data_out, rtol=1e-04, atol=1e-04
    )


if __name__ == "__main__":
    for _ in range(len(test_parameters.n_ants)):
        test_beamform_op_sequence(
            test_parameters.n_batches[0],
            test_parameters.n_ants[0],
            test_parameters.n_channels[0],
            test_parameters.n_samples_per_channel[0],
            test_parameters.n_beams[0],
            test_parameters.xeng_id[0],
            test_parameters.samples_delay[0],
            test_parameters.phase[0],
        )
