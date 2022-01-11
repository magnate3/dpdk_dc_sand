"""
Module for beamformer complex multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
import math

import numpy as np
from numba import cuda


@cuda.jit
def run_coeff_gen(
    delay_vals, batches, pols, n_channels, total_channels, n_beams, n_ants, xeng_id, sample_period, coeffs
):
    """Execute Beamforming steering coefficients."""
    # Compute flattened index inside the array
    ithreadindex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if ithreadindex_x <= (batches * pols * n_channels * n_beams * n_ants):
        # Compute indexes for delay_vals matrix
        ibatchindex_rem = ithreadindex_x % (pols * n_channels * n_beams * n_ants)

        ipolindex_rem = ibatchindex_rem % (n_channels * n_beams * n_ants)

        ichannelindex = ipolindex_rem // (n_beams * n_ants)
        ichannelindex_rem = ipolindex_rem % (n_beams * n_ants)

        iantindex = ichannelindex_rem // (n_beams)
        iantindex_rem = ichannelindex_rem % (n_beams)

        ibeamindex = iantindex_rem

        # Extract delay and phase values
        delay_s = delay_vals[ichannelindex][ibeamindex][iantindex][0]
        phase_rad = delay_vals[ichannelindex][ibeamindex][iantindex][2]

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
    """Class for beamform complex multiplication.

    Parameters
    ----------
    delay_vals:
        Data matrix of delay values.
    batches:
        Number of batches to process.
    pols:
        Number of polarisations.
    n_channels_per_stream:
        The number of channels the XEng core will process.
    total_channels:
        The total number of channels in the system.
    n_blocks:
        Number of blocks into which samples are divided in groups of 16
    samples_per_block:
        Number of samples to process per sample-block
    n_ants:
        The number of antennas from which data will be received.
    n_beams:
         Number of beams to be steered.
    xeng_id:
        Identify of the XEngine. This is used to compute the actual channel numbers processed per engine.
    sample_period:
        Sampling period of the ADC.
    """

    def __init__(
        self,
        delay_vals: np.ndarray,
        batches: int,
        num_pols: int,
        n_channels_per_stream: int,
        total_channels: int,
        n_blocks: int,
        samples_per_block: int,
        n_ants: int,
        n_beams: int,
        xeng_id: int,
        sample_period: int,
    ):
        self.delay_vals = delay_vals
        self.batches = batches
        self.pols = num_pols
        self.n_channels = n_channels_per_stream
        self.total_channels = total_channels
        self.n_blocks = n_blocks
        self.samples_per_block = samples_per_block
        self.n_ants = n_ants
        self.n_beams = n_beams
        self.xeng_id = xeng_id
        self.sample_period = sample_period
        self.total_length = self.batches * self.pols * self.n_channels * self.n_blocks * self.samples_per_block
        self.complexity = 2  # Always

    def coeff_gen(self):
        """Complex multiplication setup."""
        complexity = 2
        # The 'cols' variable is a dimensionality duplication factor required when creating the ants-beams coeff matrix.
        # The last two dimensions (ants and beams) needs to be doubled to accommodate the duplucation and negation
        # of real and imaginary coefficients in order to apply complex multiplication through real-only values.
        cols = 2

        # Temp - a place holder until this code get added into the op sequence.
        coeff_matrix = np.empty(
            self.batches * self.pols * self.n_channels * self.n_ants * self.n_beams * complexity * cols,
            dtype=np.float32,
        )
        coeff_matrix = coeff_matrix.reshape(
            self.batches, self.pols, self.n_channels, self.n_ants * complexity, self.n_beams * cols
        )

        threadsperblock = 128

        # Calculate the number of thread blocks in the grid
        blockspergrid = np.uint(
            np.ceil(self.batches * self.pols * self.n_channels * self.n_beams * self.n_ants / threadsperblock)
        )

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue. This will make the
        # context associated with the deivce device_id the current context.
        cuda.select_device(0)

        # Now start the kernel
        run_coeff_gen[blockspergrid, threadsperblock](
            self.delay_vals,
            self.batches,
            self.pols,
            self.n_channels,
            self.total_channels,
            self.n_beams,
            self.n_ants,
            self.xeng_id,
            self.sample_period,
            coeff_matrix,
        )

        # Wait for all commands in the stream to finish executing.
        cuda.synchronize()

        return coeff_matrix
