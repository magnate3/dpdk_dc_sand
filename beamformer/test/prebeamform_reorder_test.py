"""
Module for performing unit tests on the Pre-beamform Reorder.

The pre-beamform reorder operates on a block of data with the following dimensions:
    - uint16_t [n_antennas] [n_channels] [n_samples_per_channel] [polarizations]
      transposed to
      uint16_t [n_batches][polarizations][n_channels] [n_blocks][samples_per_channel//n_blocks][n_ants]
    - Typical values for the dimensions
        - n_antennas (a) = 64
        - n_channels (c) = 128
        - n_samples_per_channel (t) = 256
        - polarisations (p) = 2, always
        - n_blocks = 16, always

Contains one test (parametrised):
    1. The first test uses the list of values present in test/test_parameters.py to run the
        kernel through a range of value combinations.
        - This is limited to a batch of one, as the CPU-side verification takes some time to complete.
"""
import pytest
import test_parameters
import numpy as np
from beamform_reorder.prebeamform_reorder import PreBeamformReorderTemplate
from katsdpsigproc import accel
from beamform_reorder import reorder

# DEBUG - to be removed when debug complete
def print_mismatch(output_data_cpu, bufReordered_host):
    """Debug methos for printing differences betwen arrays with index positions."""
    if len(output_data_cpu) != len(bufReordered_host):
        print(f"output_data_cpu is {output_data_cpu} and bufReordered_host is {bufReordered_host}")
    else:
        shape = np.shape(output_data_cpu)
        batches = shape[0]
        pols = shape[1]
        n_channel = shape[2]
        samples_chan = shape[3]
        n_blocks = shape[4]
        ants = shape[5]
        count = 0

        for batch in range(batches):
            for pol in range(pols):
                for chan in range(n_channel):
                    for tbs in range(n_blocks):
                        for sample in range(samples_chan):
                            for ant in range(ants):
                                count = count + 1
                                if (
                                    output_data_cpu[batch][pol][chan][tbs][sample][ant]
                                    != bufReordered_host[batch][pol][chan][tbs][sample][ant]
                                ):
                                    print(
                                        f"output_data_cpu is {output_data_cpu[batch][pol][chan][tbs][sample][ant]}"
                                        "and bufReordered_host is {bufReordered_host[batch][pol][chan][tbs][sample][ant]}: "
                                        "batch:{batch} pol:{pol} chan:{chan} tbs:{tbs} sample:{sample} ant:{ant}"
                                    )
    print(f"Count is: {count}")


@pytest.mark.parametrize("batches", test_parameters.batches)
@pytest.mark.parametrize("num_ants", test_parameters.array_size)
@pytest.mark.parametrize("num_channels", test_parameters.num_channels)
@pytest.mark.parametrize("num_samples_per_channel", test_parameters.num_samples_per_channel)
@pytest.mark.parametrize("n_blocks", test_parameters.n_blocks)
def test_prebeamform_reorder_parametrised(batches, num_ants, num_channels, num_samples_per_channel, n_blocks):
    """
    Parametrised unit test of the Pre-beamform Reorder kernel.

    This unit test runs the kernel on a combination of parameters indicated in test_parameters.py. The values parametrised
    are indicated in the parameter list, operating on a *single* batch. This unit test also invokes verification of the reordered data.

    Parameters
    ----------
    batches: int
        Number of batches to process.
    num_ants: int
        The number of antennas from which data will be received.
    num_channels: int
        The number of frequency channels out of the FFT.
        NB: This is not the number of FFT channels per stream.
        The number of channels per stream is calculated from this value.
    num_samples_per_channel: int
        The number of time samples per frequency channel.
    n_blocks: int
        Number of blocks to break the number of samples per channel into.

    This test:
        1. Populate a host-side array with random, numpy.uint16 data ranging from 0 to 16383-2.
        2. Instantiate the prebeamformer_reorder_kernel and passes this input data to it.
        3. Grab the output, reordered data.
        4. Verify it relative to the input array.
    """
    
    # Now to create the actual PreBeamformReorderTemplate
    # 1. Array parameters
    # - Will be {ants, chans, samples_per_chan, batches}
    # - Will pass num_{ants, samples_per_channel} parameters straight into Template instantiation

    # This integer division is so that when num_ants % num_channels !=0 then the remainder will be dropped.
    # - This will only occur in the MeerKAT Extension correlator.
    # TODO: Need to consider the case where we round up as some X-Engines will need to do this to capture all the channels.
    n_channels_per_stream = num_channels // num_ants // 4

    pol = 2  # Always

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda, interactive=False)
    queue = ctx.create_command_queue()

    template = PreBeamformReorderTemplate(
        ctx,
        n_ants=num_ants,
        n_channels=n_channels_per_stream,
        n_samples_per_channel=num_samples_per_channel,
        n_batches=batches,
    )
    preBeamformReorder = template.instantiate(queue)
    preBeamformReorder.ensure_all_bound()

    bufSamples_device = preBeamformReorder.buffer("inSamples")
    bufSamples_host = bufSamples_device.empty_like()

    bufReordered_device = preBeamformReorder.buffer("outReordered")
    bufReordered_host = bufReordered_device.empty_like()

    # 3. Generate random input data - need to modify the dtype and shape of the array as numpy does not have a packet
    # 8-bit int complex type.

    # DEBUG: Inject sample to trace
    # bufSamples_host.dtype = np.uint8
    # bufSamples_host[:] = np.zeros(bufSamplesInt8Shape, dtype=np.uint8)
    # bufSamples_host[0][0][0][0][0] = 1 #Pol0 Sample0 Real
    # bufSamples_host[0][0][0][0][1] = 55 #Pol0 Sample0 Imag

    # bufSamples_host[0][0][0][1][0] = 77 #Pol0 Sample1 Real
    # bufSamples_host[0][0][0][1][1] = 33 #Pol0 Sample1 Imag

    # OR

    # Inject random data for test.
    rng = np.random.default_rng(seed=2021)

    bufSamples_host[:] = rng.uniform(
        np.iinfo(bufSamples_host.dtype).min, np.iinfo(bufSamples_host.dtype).max, bufSamples_host.shape
    ).astype(bufSamples_host.dtype)

    # 4. Transfer input sample array to the GPU, run kernel, transfer output Reordered array to the CPU.
    bufSamples_device.set(queue, bufSamples_host)
    preBeamformReorder()
    bufReordered_device.get(queue, bufReordered_host)

    # 5. Run CPU version. This will be used to verify GPU reorder.
    output_data_cpu = reorder.reorder(
        input_data=bufSamples_host,
        input_data_shape=bufSamples_host.shape,
        output_data_shape=bufReordered_host.shape,
    )

    # DEBUG: Print out differences (with location)
    # print_mismatch(output_data_cpu, bufReordered_host)

    # 6. Verify the processed/returned result
    #    - Both the input and output data are ultimately of type np.int8

    # DEBUG: Force failure.
    # output_data_cpu[0][0][0][0][0][0] = 9

    # 7. Check if both arrays are identical.
    # These views will bail if memory is not contiguous.
    #viewed_cpu = output_data_cpu.view(dtype=np.int8)
    #viewed_gpu = bufReordered_host.view(dtype=np.int8)
    #np.testing.assert_array_equal(viewed_cpu, viewed_gpu)
    np.testing.assert_array_equal(output_data_cpu, bufReordered_host)

if __name__ == "__main__":
    for a in range(len(test_parameters.array_size)):
        test_prebeamform_reorder_parametrised(
            test_parameters.batches[0],
            test_parameters.array_size[a],
            test_parameters.num_channels[0],
            test_parameters.num_samples_per_channel[0],
            test_parameters.n_blocks[0],
        )
