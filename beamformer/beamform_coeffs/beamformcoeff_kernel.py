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
    # # Compute flattened index inside the array
    iThreadIndex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Compute indexes for delay_vals matrix
    iBatchIndex = iThreadIndex_x // (pols * n_channels * n_beams * n_ants)
    iBatchIndex_rem = iThreadIndex_x % (pols * n_channels * n_beams * n_ants)

    iPolIndex = iBatchIndex_rem // (n_channels * n_beams * n_ants)
    iPolIndex_rem = iBatchIndex_rem % (n_channels * n_beams * n_ants)

    iChannelIndex = iPolIndex_rem // (n_beams * n_ants)
    iChannelIndex_rem = iPolIndex_rem % (n_beams * n_ants)

    iAntIndex = iChannelIndex_rem // (n_beams)
    iAntIndex_rem = iChannelIndex_rem % (n_beams)

    iBeamIndex = iAntIndex_rem

    # Extract delay and phase values
    Delay_s = delay_vals[iChannelIndex][iBeamIndex][iAntIndex][0]
    DelayRate_sps = delay_vals[iChannelIndex][iBeamIndex][iAntIndex][1]
    Phase_rad = delay_vals[iChannelIndex][iBeamIndex][iAntIndex][2]
    PhaseRate_radps = delay_vals[iChannelIndex][iBeamIndex][iAntIndex][3]

    # Compute actual channel index (i.e. channel in spectrum being computed on)
    # This is needed when computing the rotation value before the cos/sin lookup.
    # There are n_channels per xeng so adding n_channels * xeng_id gives the
    # relative channel in the spectrum the xeng GPU thread is working on.
    iChannel = iChannelIndex + n_channels * xeng_id

    initial_phase = Delay_s * iChannel * (-np.math.pi) / (total_channels * sample_period) + Phase_rad

    Phase_correction_band_center = Delay_s * (total_channels / 2) * (-np.math.pi) / (total_channels * sample_period)

    # Compute rotation value for steering coefficient computation
    Rotation = initial_phase - Phase_correction_band_center

    SteeringCoeffCorrectReal = math.cos(Rotation)
    SteeringCoeffCorrectImag = math.sin(Rotation)

    # Compute indexes for output matrix
    iBatchIndex = iThreadIndex_x // (pols * n_channels * n_beams * n_ants)
    iBatchIndex_rem = iThreadIndex_x % (pols * n_channels * n_beams * n_ants)

    iPolIndex = iBatchIndex_rem // (n_channels * n_beams * n_ants)
    iPolIndex_rem = iBatchIndex_rem % (n_channels * n_beams * n_ants)

    iChannelIndex = iPolIndex_rem // (n_beams * n_ants)
    iChannelIndex_rem = iPolIndex_rem % (n_beams * n_ants)

    iBeamIndex = iChannelIndex_rem // (n_ants)
    iBeamIndex_rem = iChannelIndex_rem % (n_ants)

    iBeamMatrix = iBeamIndex * 2
    iAntMatrix = iBeamIndex_rem * 2

    # Store steering coefficients in output matrix
    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix][iBeamMatrix] = SteeringCoeffCorrectReal
    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix][iBeamMatrix + 1] = SteeringCoeffCorrectImag

    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix + 1][iBeamMatrix] = -SteeringCoeffCorrectImag
    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix + 1][iBeamMatrix + 1] = SteeringCoeffCorrectReal


class beamform_coeff_kernel:
    """Class for beamform complex multiplication."""

    def coeff_gen(delay_vals, batches, pols, n_beams, num_channels, total_channels, n_ants, xeng_id, sample_period):
        """Complex multiplication setup.

        Parameters
        ----------
        delay_vals: nd.array[float (single)]
            Data matrix of delay values.
        batches: int
            Number of batches to process.
        pols: int
            Number of polarisations.
        n_beams: int
             Number of beams to be steered.
        num_channels: int
            The number of channels the XEng core will process.
        total_channels: int
            The total number of channels in the system.
        n_ants: int
            The number of antennas from which data will be received.
        xeng_id: int
            Identify of the XEngine. This is used to compute the actual channel numbers processed per engine.
        sample_period: int
            Sampling period of the ADC.
        """
        complexity = 2
        cols = 2

        # Temp - a place holder until this code get added into the op sequence.
        coeff_matrix = np.empty(batches * pols * num_channels * n_ants * n_beams * complexity * cols, dtype=np.float32)
        coeff_matrix = coeff_matrix.reshape(batches, pols, num_channels, n_ants * complexity, n_beams * cols)

        # Calculate the number of threads required.
        total_threads = batches * pols * num_channels * n_beams * n_ants

        # Calculate the largest divisor possible (withing range 2-1024)
        largest_divisor = 0
        for i in range(2, total_threads):
            if (total_threads % i == 0) & (i > 1 & i <= 1024):
                largest_divisor = i
            elif i > 1024:
                break

        threadsperblock = largest_divisor

        # Calculate the number of thread blocks in the grid
        blockspergrid = batches * pols * num_channels * n_beams * n_ants // threadsperblock

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue. This will make the
        # context associated with the deivce device_id the current context.
        cuda.select_device(0)

        # Now start the kernel
        run_coeff_gen[blockspergrid, threadsperblock](
            delay_vals, pols, num_channels, total_channels, n_beams, n_ants, xeng_id, sample_period, coeff_matrix
        )

        # Wait for all commands in the stream to finish executing.
        cuda.synchronize()

        return coeff_matrix
