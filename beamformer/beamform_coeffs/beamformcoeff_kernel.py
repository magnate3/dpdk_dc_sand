"""
Module for beamformer complex multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
from numba import cuda, float32
import math
import numpy as np

@cuda.jit
def run_coeff_gen(delay_vals, batches, pols, n_channels, total_channels, n_beams, n_ants, xeng_id, sample_period, coeffs):
    """Execute complex multiplication.
    """
    # # Compute flattened index inside the array
    iThreadIndex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # start = 2 * (2 * (36 * 7 * 2 * 8 * 2)) + 1 * (36 * 7 * 2 * 8 * 2) + (34 * 7 * 2 * 8 * 2)*0
    # debug_thread_idx_lower =  2 * (2 * (36 * 7 * 8)) + 1 * (36 * 7 * 8) + (34 * 7 * 8) -16
    debug_thread_idx_lower = 0
    debug_thread_idx_upper = debug_thread_idx_lower + 64

    # if iThreadIndex_x == debug_thread_idx:
    #     print('iThreadIndex_x', iThreadIndex_x)
    #     print('cuda.blockIdx.x',cuda.blockIdx.x)
    #     print('cuda.blockDim.x',cuda.blockDim.x)

    # Compute indexes for delay_vals matrix
    iBatchIndex = iThreadIndex_x // (pols*n_channels*n_beams*n_ants)
    iBatchIndex_rem = iThreadIndex_x % (pols*n_channels*n_beams*n_ants)

    iPolIndex = iBatchIndex_rem // (n_channels*n_beams * n_ants)
    iPolIndex_rem = iBatchIndex_rem % (n_channels*n_beams * n_ants)

    iChannelIndex = iPolIndex_rem // (n_beams*n_ants)
    iChannelIndex_rem = iPolIndex_rem % (n_beams*n_ants)

    iAntIndex = iChannelIndex_rem // (n_beams)
    iAntIndex_rem = iChannelIndex_rem % (n_beams)

    iBeamIndex = iAntIndex_rem

    # if (iThreadIndex_x > debug_thread_idx_lower):
    #     print('thread:', iThreadIndex_x,
    #         'iBatchIndex', iBatchIndex,
    #         'iBatchIndex_rem', iBatchIndex_rem,
    #         'iPolIndex:', iPolIndex,
    #         'iPolIndex_rem:', iPolIndex_rem,
    #         'iChanIndex:', iChannelIndex,
    #         'iChannelIndex_rem:', iChannelIndex_rem,
    #         'iAntIndex:', iAntIndex,
    #         'iAntIndex_rem:', iAntIndex_rem,
    #         'iBeamIndex:', iBeamIndex)

    # Extract delay and phase values
    Delay_s = delay_vals[iChannelIndex][iBeamIndex][iAntIndex][0]
    DelayRate_sps = delay_vals[iChannelIndex][iBeamIndex][iAntIndex][1]
    Phase_rad = delay_vals[iChannelIndex][iBeamIndex][iAntIndex][2]
    PhaseRate_radps = delay_vals[iChannelIndex][iBeamIndex][iAntIndex][3]

    # Compute rotation value for steering coefficient computation

    # Compute actual channel index (i.e. channel in spectrum being computed on)
    # This is needed when computing the rotation value before the cos/sin lookup.
    # There are n_channels per xeng so adding n_channels * xeng_id gives the
    # relative channel in the spectrum the xeng GPU thread is working on.
    # iChannel = iChannelIndex//(batches*pols) + n_channels * xeng_id
    iChannel = iChannelIndex + n_channels * xeng_id

    initial_phase = Delay_s * iChannel * (-np.math.pi) / (total_channels * sample_period) + Phase_rad

    Phase_correction_band_center = Delay_s * (total_channels/2) * (-np.math.pi) / (total_channels * sample_period)

    Rotation = initial_phase - Phase_correction_band_center

    SteeringCoeffCorrectReal = math.cos(Rotation)
    SteeringCoeffCorrectImag = math.sin(Rotation)

    # Compute indexes for output matrix
    iBatchIndex = iThreadIndex_x // (pols*n_channels*n_beams*n_ants)
    iBatchIndex_rem = iThreadIndex_x % (pols*n_channels*n_beams*n_ants)

    iPolIndex = iBatchIndex_rem // (n_channels*n_beams * n_ants)
    iPolIndex_rem = iBatchIndex_rem % (n_channels*n_beams * n_ants)

    iChannelIndex = iPolIndex_rem // (n_beams*n_ants)
    iChannelIndex_rem = iPolIndex_rem % (n_beams*n_ants)

    iBeamIndex = iChannelIndex_rem // (n_ants)
    iBeamIndex_rem = iChannelIndex_rem % (n_ants)

    iBeamMatrix = iBeamIndex*2
    iAntMatrix = iBeamIndex_rem*2

    # if(iThreadIndex_x>11999):
    #     print('iThreadIndex_x',iThreadIndex_x)

    # if (iThreadIndex_x > debug_thread_idx_lower)&(iThreadIndex_x < debug_thread_idx_upper):
    #     print('thread:', iThreadIndex_x,
    #         'iBatchIndex', iBatchIndex,
    #         'iBatchIndex_rem', iBatchIndex_rem,
    #         'iPolIndex:', iPolIndex,
    #         'iPolIndex_rem:', iPolIndex_rem,
    #         'iChanIndex:', iChannelIndex,
    #         'iChannelIndex_rem:', iChannelIndex_rem,
    #         'iBeamIndex:', iBeamIndex,
    #         'iBeamIndex_rem:', iBeamIndex_rem,
    #         'iBeamMatrix:', iBeamMatrix,
    #         'iAntMatrix:', iAntMatrix)

    # if iAntMatrix > 12:
    #     print('iAntMatrix', iAntMatrix)

    # if ((iBatchIndex == 0) & (iPolIndex == 0) & (iChannelIndex == 0)& (iAntMatrix == 0)):
    #     print('iThreadIndex_x', iThreadIndex_x, 'iBeamMatrix', iBeamMatrix)

    # Store steering coefficients in output matrix
    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix][iBeamMatrix] = SteeringCoeffCorrectReal #4
    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix][iBeamMatrix+1] = SteeringCoeffCorrectImag #1

    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix+1][iBeamMatrix] = -SteeringCoeffCorrectImag #-1
    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix+1][iBeamMatrix+1] = SteeringCoeffCorrectReal #4

    # if iThreadIndex_x > 0:
    #     coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix][iBeamMatrix] = SteeringCoeffCorrectReal #4
    #     coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix][iBeamMatrix+1] = SteeringCoeffCorrectImag #1

    #     coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix+1][iBeamMatrix] = -SteeringCoeffCorrectImag #-1
    #     coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix+1][iBeamMatrix+1] = SteeringCoeffCorrectReal #4

class beamform_coeff_kernel:
    """Class for beamform complex multiplication."""

    def coeff_gen(delay_vals, batches, pols, n_beams, num_channels, total_channels, n_ants, xeng_id, sample_period):
    # def coeff_gen(self, current_time, ref_time, delay, coeff_matrix):
        """Complex multiplication setup.

        Parameters
        ----------
        data_matrix: nd.array[np.uint8]
            Data matrix on reordered data
        coeff_matrix: nd.array[np.float32]
            Coefficients for beamforming computation.
        out: nd.array[np.float32]
            Complex multipication product for beamforming computation.
        """
        complexity = 2
        cols = 2

        # Temp
        coeff_matrix = np.empty(batches*pols*num_channels*n_ants*n_beams*complexity*cols, dtype=np.float32)
        coeff_matrix = coeff_matrix.reshape(batches, pols, num_channels, n_ants*complexity, n_beams*cols)

        # Set the number of threads in a block
        # threadsperblock = 128

        largest_divisor = 0
        num = batches*pols*num_channels*n_beams*n_ants
        for i in range(2, num):
            if ((num % i == 0) & (i<=1024)):
                largest_divisor = i
            elif (i > 1024):
                break

        threadsperblock = largest_divisor
        
        # Calculate the number of thread blocks in the grid
        blockspergrid = batches*pols*num_channels*n_beams*n_ants // threadsperblock

        # blockspergrid = int(np.ceil((batches * pols * num_channels * n_beams * n_ants) / threadsperblock))

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue. This will make the
        # context associated with the deivce device_id the current context.
        cuda.select_device(0)

        # Now start the kernel
        run_coeff_gen[blockspergrid, threadsperblock](delay_vals, batches, pols, num_channels, total_channels, n_beams, n_ants, xeng_id, sample_period, coeff_matrix)

        # Wait for all commands in the stream to finish executing.
        cuda.synchronize()

        return coeff_matrix
