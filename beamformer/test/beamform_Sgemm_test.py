"""
Module for performing unit tests on the beamform operation.

The beamform operation occurs on a reordered block of data with the following dimensions:
    - uint16_t [n_batches][polarizations][n_channels][n_blocks][samples_per_block][n_ants][complexity]

Contains one test (parametrised):
    1. 
"""

import numpy as np
import pytest
import test_parameters
from beamform_reorder import reorder
from beamform_reorder.prebeamform_reorder import PreBeamformReorderTemplate
# from beamforming.beamform import MultiplyTemplate
from katsdpsigproc import accel
from beamforming import beamform
from beamforming import complex_mult_cpu
from numba import jit

class CoeffGenerator:
    def __init__(self, batches, pols, num_chan, n_blocks, samples_per_block, ants):
        self.batches = batches
        self.pols = pols
        self.num_chan = num_chan
        self.n_blocks = n_blocks
        self.samples_per_block = samples_per_block
        self.ants = ants
        self.total_length = self.batches * self.pols * self.num_chan * self.n_blocks * self.samples_per_block
    
    @jit
    def GPU_Coeffs(self):
        # coeffs = np.arange(1,((ants*2)*2*l+1),1,np.float32).reshape(l,2,ants * 2)
        coeffs = np.ones(((self.ants*2)*2*self.total_length),np.float32).reshape(self.total_length,2,self.ants * 2)
        real_value = 4
        imag_value = 1
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                for k in range(coeffs.shape[2]):
                    if j == 0:
                        if k % 2:
                            coeffs[i,j,k] = -1 * real_value
                        else:
                            coeffs[i,j,k] = imag_value
                    else:
                        if k % 2:
                            coeffs[i,j,k] = imag_value
                        else:
                            coeffs[i,j,k] = real_value
        return coeffs

    @jit
    def CPU_Coeffs(self):
        # coeffs = np.arange(1,((ants*2)*2*l+1),1,np.float32).reshape(l,2,ants * 2)
        # coeffs = np.ones(((self.ants)*2*self.total_length),np.float32).reshape(self.total_length,self.ants * 2)
        coeffs = np.ones(self.ants*2*self.total_length,np.float32).reshape(self.total_length,self.ants, 2)

        real_value = 4
        imag_value = 1
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                for k in range(coeffs.shape[2]):
                    if k == 0:
                        coeffs[i,j,k] = real_value
                    else:
                        coeffs[i,j,k] = imag_value
        return coeffs.reshape(self.batches, self.pols, self.num_chan, self.n_blocks, self.samples_per_block, self.ants, 2)

@pytest.mark.parametrize("batches", test_parameters.batches)
@pytest.mark.parametrize("num_ants", test_parameters.array_size)
@pytest.mark.parametrize("num_channels", test_parameters.num_channels)
@pytest.mark.parametrize("num_samples_per_channel", test_parameters.num_samples_per_channel)
def test_beamform_parametrised(batches, num_ants, num_channels, num_samples_per_channel):
    """
    Parametrised unit test of the beamform kernel.

    This unit test runs the kernel on a combination of parameters indicated in test_parameters.py. The values
    parametrised are indicated in the parameter list, operating on a *single* batch. This unit test also invokes
    verification of the reordered data.

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
    n_samples_per_block: int
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
    n_channels_per_stream = num_channels // num_ants // 4
    pols = 2
    samples_per_block = 16
    n_blocks = num_samples_per_channel // samples_per_block
    coeff_gen = CoeffGenerator(batches, pols, n_channels_per_stream, n_blocks, samples_per_block, num_ants)
    gpu_coeffs = coeff_gen.GPU_Coeffs()

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda, interactive=False)
    queue = ctx.create_command_queue()

    # Create BeamformTemplate and link to buffer slots
    beamform_mult_template = beamform.MultiplyTemplate(
        ctx,
        n_ants=num_ants,
        n_channels=n_channels_per_stream,
        n_samples_per_channel=num_samples_per_channel,
        n_batches=batches,
    )

    BeamformMult = beamform_mult_template.instantiate(queue, gpu_coeffs)
    BeamformMult.ensure_all_bound()

    bufSamples_device = BeamformMult.buffer("inData")
    bufSamples_host = bufSamples_device.empty_like()

    bufBeamform_device = BeamformMult.buffer("outData")
    bufBeamform_host = bufBeamform_device.empty_like()

    # 3. Generate random input data - need to modify the dtype and shape of the array as numpy does not have a packet
    # 8-bit int complex type.

    # Inject random data for test.
    rng = np.random.default_rng(seed=2021)

    bufSamples_host[:] = rng.uniform(
        np.iinfo(bufSamples_host.dtype).min, np.iinfo(bufSamples_host.dtype).max, bufSamples_host.shape
    ).astype(bufSamples_host.dtype)

    # bufSamples_host[:] = np.ones(bufSamples_host.shape,np.float32)

    # 4. Reorder: Transfer input sample array to the GPU, run reorder kernel, transfer output Reordered array to the CPU.
    bufSamples_device.set(queue, bufSamples_host)
    BeamformMult()
    bufBeamform_device.get(queue, bufBeamform_host)

    # 5. Run CPU version. This will be used to verify GPU reorder.
    cpu_coeffs = coeff_gen.CPU_Coeffs()
    output_data_cpu = complex_mult_cpu.complex_mult(
        # input_data=bufSamples_host.reshape(batches, pols, n_channels_per_stream, n_blocks, samples_per_block, num_ants*2),
        input_data=bufSamples_host,
        coeffs=cpu_coeffs,
        output_data_shape=bufBeamform_host.shape,
    )

    # 6. Verify the processed/returned result
    #    - Both the input and output data are ultimately of type np.int8
    np.testing.assert_array_equal(output_data_cpu, bufBeamform_host)

if __name__ == "__main__":
    for a in range(len(test_parameters.array_size)):
        test_beamform_parametrised(
            test_parameters.batches[0],
            test_parameters.array_size[a],
            test_parameters.num_channels[0],
            test_parameters.num_samples_per_channel[0],
        )