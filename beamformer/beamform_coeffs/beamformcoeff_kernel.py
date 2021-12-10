"""
Module for beamformer complex multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
import math

import numpy as np
from numba import cuda


@cuda.jit
def run_coeff_gen(delay_vals, pols, n_channels, total_channels, n_beams, n_ants, xeng_id, sample_period, coeffs):
    """Execute Beamforming steering coefficients."""
    # Compute flattened index inside the array
    ithreadindex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Compute indexes for delay_vals matrix
    # ibatchindex = ithreadindex_x // (pols * n_channels * n_beams * n_ants)
    ibatchindex_rem = ithreadindex_x % (pols * n_channels * n_beams * n_ants)

    # ipolindex = ibatchindex_rem // (n_channels * n_beams * n_ants)
    ipolindex_rem = ibatchindex_rem % (n_channels * n_beams * n_ants)

    ichannelindex = ipolindex_rem // (n_beams * n_ants)
    ichannelindex_rem = ipolindex_rem % (n_beams * n_ants)

    iantindex = ichannelindex_rem // (n_beams)
    iantindex_rem = ichannelindex_rem % (n_beams)

    ibeamindex = iantindex_rem

    # Extract delay and phase values
    delay_s = delay_vals[ichannelindex][ibeamindex][iantindex][0]
    # DelayRate_sps = delay_vals[ichannelindex][ibeamindex][iantindex][1]
    phase_rad = delay_vals[ichannelindex][ibeamindex][iantindex][2]
    # PhaseRate_radps = delay_vals[ichannelindex][ibeamindex][iantindex][3]

    # Compute actual channel index (i.e. channel in spectrum being computed on)
    # This is needed when computing the rotation value before the cos/sin lookup.
    # There are n_channels per xeng so adding n_channels * xeng_id gives the
    # relative channel in the spectrum the xeng GPU thread is working on.
    ichannel = ichannelindex + n_channels * xeng_id

    initial_phase = delay_s * ichannel * (-np.math.pi) / (total_channels * sample_period) + phase_rad

    phase_correction_band_center = delay_s * (total_channels / 2) * (-np.math.pi) / (total_channels * sample_period)

    # Compute rotation value for steering coefficient computation
    rotation = initial_phase - phase_correction_band_center

    steering_coeff_correct_real = math.cos(rotation)
    steering_coeff_correct_imag = math.sin(rotation)

    # Compute indexes for output matrix
    ibatchindex = ithreadindex_x // (pols * n_channels * n_beams * n_ants)
    ibatchindex_rem = ithreadindex_x % (pols * n_channels * n_beams * n_ants)

    ipolindex = ibatchindex_rem // (n_channels * n_beams * n_ants)
    ipolindex_rem = ibatchindex_rem % (n_channels * n_beams * n_ants)

    ichannelindex = ipolindex_rem // (n_beams * n_ants)
    ichannelindex_rem = ipolindex_rem % (n_beams * n_ants)

    ibeamindex = ichannelindex_rem // (n_ants)
    ibeamindex_rem = ichannelindex_rem % (n_ants)

    ibeam_matrix = ibeamindex * 2
    iant_matrix = ibeamindex_rem * 2

    # Store steering coefficients in output matrix
    coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix][ibeam_matrix] = steering_coeff_correct_real
    coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix][ibeam_matrix + 1] = steering_coeff_correct_imag

    coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix + 1][ibeam_matrix] = -steering_coeff_correct_imag
    coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix + 1][ibeam_matrix + 1] = steering_coeff_correct_real


class BeamformCoeffKernel:
    """Class for beamform complex multiplication."""

    # def __init__(self, batches, n_channels, n_blocks, samples_per_block, n_ants, xeng_id):

    def __init__(
        self,
        delay_vals,
        batches,
        num_pols,
        n_channels_per_stream,
        total_channels,
        n_blocks,
        samples_per_block,
        n_ants,
        num_beams,
        xeng_id,
        sample_period,
    ):
        """Initialise the coefficient generation class."""
        self.delay_vals = delay_vals
        self.batches = batches
        self.pols = num_pols
        self.num_channels = n_channels_per_stream
        self.total_channels = total_channels
        self.n_blocks = n_blocks
        self.samples_per_block = samples_per_block
        self.n_ants = n_ants
        self.num_beams = num_beams
        self.xeng_id = xeng_id
        self.sample_period = sample_period
        self.total_length = self.batches * self.pols * self.num_channels * self.n_blocks * self.samples_per_block
        self.complexity = 2  # Always

    def coeff_gen(self):
        """Complex multiplication setup.

        Parameters
        ----------
        delay_vals: nd.array[float (single)]
            Data matrix of delay values.
        batches: int
            Number of batches to process.
        pols: int
            Number of polarisations.
        num_channels: int
            The number of channels the XEng core will process.
        total_channels: int
            The total number of channels in the system.
        n_blocks: int
            Number of blocks into which samples are divided in groups of 16
        samples_per_block: int
            Number of samples to process per sample-block
        n_ants: int
            The number of antennas from which data will be received.
        num_beams: int
             Number of beams to be steered.
        xeng_id: int
            Identify of the XEngine. This is used to compute the actual channel numbers processed per engine.
        sample_period: int
            Sampling period of the ADC.
        """
        complexity = 2
        cols = 2

        # Temp - a place holder until this code get added into the op sequence.
        coeff_matrix = np.empty(
            self.batches * self.pols * self.num_channels * self.n_ants * self.num_beams * complexity * cols,
            dtype=np.float32,
        )
        coeff_matrix = coeff_matrix.reshape(
            self.batches, self.pols, self.num_channels, self.n_ants * complexity, self.num_beams * cols
        )

        # Calculate the number of threads required.
        total_threads = self.batches * self.pols * self.num_channels * self.num_beams * self.n_ants

        # Calculate the largest divisor possible (withing range 2-1024)
        largest_divisor = 0
        for i in range(2, total_threads):
            if (total_threads % i == 0) & (i > 1 & i <= 1024):
                largest_divisor = i
            elif i > 1024:
                break

        threadsperblock = largest_divisor

        # Calculate the number of thread blocks in the grid
        blockspergrid = self.batches * self.pols * self.num_channels * self.num_beams * self.n_ants // threadsperblock

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue. This will make the
        # context associated with the deivce device_id the current context.
        cuda.select_device(0)

        # Now start the kernel
        run_coeff_gen[blockspergrid, threadsperblock](
            self.delay_vals,
            self.pols,
            self.num_channels,
            self.total_channels,
            self.num_beams,
            self.n_ants,
            self.xeng_id,
            self.sample_period,
            coeff_matrix,
        )

        # Wait for all commands in the stream to finish executing.
        cuda.synchronize()

        return coeff_matrix
