"""
Module for performing unit tests on the Pre-beamform Reorder.

The pre-beamform reorder operates on a block of data with the following dimensions:
    - uint16_t [n_antennas][n_channels][n_samples_per_channel][polarizations][complexity]
      transposed to
      uint16_t [n_batches][polarizations][n_channels][n_blocks][samples_per_block][n_ants][complexity]
    - Typical values for the dimensions
        - n_antennas (a) = 64
        - n_channels (c) = 128
        - n_samples_per_channel (t) = 256
        - polarisations (p) = 2, always
        - n_blocks = 16, always
        - samples_per_block calculated as n_samples_per_channel // n_blocks

Contains one test (parametrised):
    1. The first test uses the list of values present in test/test_parameters.py to run the
        kernel through a range of value combinations. See docstring in `test_prebeamform_reorder_parametrised`
"""

import numpy as np
import pytest
from beamform_reorder import reorder
from beamform_reorder.prebeamform_reorder import PreBeamformReorderTemplate
from katsdpsigproc import accel
from unit_test import test_parameters


@pytest.mark.parametrize("batches", test_parameters.batches)
@pytest.mark.parametrize("n_ants", test_parameters.array_size)
@pytest.mark.parametrize("n_channels", test_parameters.n_channels)
@pytest.mark.parametrize("n_samples_per_channel", test_parameters.n_samples_per_channel)
def test_prebeamform_reorder_parametrised(batches: int, n_ants: int, n_channels: int, n_samples_per_channel: int):
    """
    Parametrised unit test of the Pre-beamform Reorder kernel.

    This unit test runs the kernel on a combination of parameters indicated in test_parameters.py. The values
    parametrised are indicated in the parameter list, operating on a *single* batch. This unit test also invokes
    verification of the reordered data.

    Parameters
    ----------
    batches:
        Number of batches to process.
    num_ants:
        The number of antennas from which data will be received.
    num_channels:
        The number of frequency channels out of the FFT.
        NB: This is not the number of FFT channels per stream.
        The number of channels per stream is calculated from this value.
    num_samples_per_channel:
        The number of time samples per frequency channel.
    n_samples_per_block:
        Number of samples per block.

    This test:
        1. Populate a host-side array with random data in the range of the relevant dtype.
        2. Instantiate the prebeamformer_reorder_kernel and pass this input data to it.
        3. Grab the output, reordered data.
        4. Verify it relative to the input array using a reference computed on the host.
    """
    # Now to create the actual PreBeamformReorderTemplate
    # 1. Array parameters
    # - Will be {ants, chans, samples_per_chan, batches}
    # - Will pass num_{ants, samples_per_channel} parameters straight into Template instantiation

    # This integer division is so that when num_ants % num_channels !=0 then the remainder will be dropped.
    # - This will only occur in the MeerKAT Extension correlator.
    # TODO: Need to consider case where we round up as some X-Engines will need to do this to capture all the channels.
    n_channels_per_stream = n_channels // n_ants // 4

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda, interactive=False)
    queue = ctx.create_command_queue()

    template = PreBeamformReorderTemplate(
        ctx,
        n_ants=n_ants,
        n_channels=n_channels_per_stream,
        n_samples_per_channel=n_samples_per_channel,
        n_batches=batches,
    )
    pre_beamform_reorder = template.instantiate(queue)
    pre_beamform_reorder.ensure_all_bound()

    buf_samples_device = pre_beamform_reorder.buffer("inSamples")
    buf_samples_host = buf_samples_device.empty_like()

    buf_reordered_device = pre_beamform_reorder.buffer("outReordered")
    buf_reordered_host = buf_reordered_device.empty_like()

    # 3. Generate random input data - need to modify the dtype and shape of the array as numpy does not have a packet
    # 8-bit int complex type.

    # Inject random data for test.
    rng = np.random.default_rng(seed=2021)

    buf_samples_host[:] = rng.uniform(
        np.iinfo(buf_samples_host.dtype).min, np.iinfo(buf_samples_host.dtype).max, buf_samples_host.shape
    ).astype(buf_samples_host.dtype)

    # 4. Transfer input sample array to the GPU, run kernel, transfer output Reordered array to the CPU.
    buf_samples_device.set(queue, buf_samples_host)
    pre_beamform_reorder()
    buf_reordered_device.get(queue, buf_reordered_host)

    # 5. Run CPU version. This will be used to verify GPU reorder.
    output_data_cpu = reorder.reorder(
        input_data=buf_samples_host,
        input_data_shape=buf_samples_host.shape,
        output_data_shape=buf_reordered_host.shape,
    )

    # 6. Verify the processed/returned result
    #    - Both the input and output data are ultimately of type np.int8
    np.testing.assert_array_equal(output_data_cpu, buf_reordered_host)


if __name__ == "__main__":
    for a in range(len(test_parameters.array_size)):
        test_prebeamform_reorder_parametrised(
            test_parameters.batches[0],
            test_parameters.array_size[a],
            test_parameters.n_channels[0],
            test_parameters.n_samples_per_channel[0],
        )
