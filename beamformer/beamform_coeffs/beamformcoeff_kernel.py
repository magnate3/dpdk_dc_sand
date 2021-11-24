"""
Module for beamformer complex multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
from numba import cuda, float32
import math
import numpy as np

@cuda.jit
def run_coeff_gen(current_time_sec, ref_time_sec, delay_vals, batches, pols, n_channels, n_beams, n_ants, coeffs):
    # def run_coeff_gen(current_time, ref_time, delay, n_channel, n_beams, n_ants, coeff_matrix):

    """Execute complex multiplication.
    """
    # # Compute flattened index inside the array
    iThreadIndex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    debug_thread_idx = 4095

    # if iThreadIndex_x == debug_thread_idx:
    #     print('iThreadIndex_x', iThreadIndex_x)
    #     print('cuda.blockIdx.x',cuda.blockIdx.x)
    #     print('cuda.blockDim.x',cuda.blockDim.x)

    current_time_ns = current_time_sec
    ref_time_ns = ref_time_sec
    SAMPLING_PERIOD = 1e-7
    complexity = 2
    # duplicate = 2

    iChannelIndex = iThreadIndex_x // (n_beams*n_ants)
    iChannelIndex_rem = iThreadIndex_x % (n_beams*n_ants)

    iAntIndex = iChannelIndex_rem // (n_beams)
    iAntIndex_rem = iChannelIndex_rem % (n_beams)

    iBeamIndex = iAntIndex_rem

    Delay_s = delay_vals[iChannelIndex][iAntIndex][iBeamIndex][0]
    DelayRate_sps = delay_vals[iBeamIndex][iChannelIndex][iAntIndex][1]
    Phase_rad = delay_vals[iBeamIndex][iChannelIndex][iAntIndex][2]
    PhaseRate_radps = delay_vals[iBeamIndex][iChannelIndex][iAntIndex][3]

    # if iThreadIndex_x == debug_thread_idx:
    #     print('iChannelIndex:', iChannelIndex)
    #     print('iChannelIndex_rem:', iChannelIndex_rem)
    #     print('iAntIndex:', iAntIndex)
    #     print('iAntIndex_rem:', iAntIndex_rem)
    #     print('iBeamIndex:', iBeamIndex)
    #
    #     print('Delay_s:', Delay_s)
    #     print('DelayRate_sps', DelayRate_sps)
    #     print('Phase_rad', Phase_rad)
    #     print('PhaseRate_radps', PhaseRate_radps)

    TimeDifference = current_time_sec - ref_time_sec
    NanosecondsTimeDifference = current_time_ns - ref_time_ns
    TimeDifference += NanosecondsTimeDifference/1e9

    DeltaTime = TimeDifference
    DeltaDelay = DelayRate_sps*DeltaTime
    DelayN = (DelayRate_sps + DeltaDelay) * iChannelIndex * (np.pi) / (
                    SAMPLING_PERIOD * n_channels)
    DelayN2 = (Delay_s + DeltaDelay)*(n_channels/2)*(np.pi)/(SAMPLING_PERIOD*n_channels)
    DeltaPhase = PhaseRate_radps * DeltaTime
    Phase0 = Phase_rad - DelayN2 + DeltaPhase
    Rotation = DelayN + Phase0

    SteeringCoeffCorrectReal = math.cos(Rotation)
    SteeringCoeffCorrectImag = math.sin(Rotation)

    if iThreadIndex_x == debug_thread_idx:
        print('Steering Real:', SteeringCoeffCorrectReal)
        print('Steering Imag:', SteeringCoeffCorrectImag)

    # Compute indexes for output matrix
    iBatchIndex = iThreadIndex_x // (pols*n_channels*n_beams*2*n_ants*2)
    iBatchIndex_rem = iThreadIndex_x % (pols*n_channels*n_beams*2*n_ants*2)

    iPolIndex = iBatchIndex_rem // (n_channels*n_beams*2 * n_ants*2)
    iPolIndex_rem = iBatchIndex_rem % (n_channels*n_beams*2 * n_ants*2)

    iChannelIndex = iPolIndex_rem // (n_beams*2*n_ants*2)
    iChannelIndex_rem = iPolIndex_rem % (n_beams*2*n_ants*2)

    iAntIndex = iChannelIndex_rem // (n_beams*2*2)
    iAntIndex_rem = iChannelIndex_rem % (n_beams*2*2)

    iAntMatrix = iAntIndex*2

    iBeamIndex = iAntIndex_rem//(2*2)
    iBeamMatrix = iBeamIndex*2

    # if iThreadIndex_x == debug_thread_idx:
    #     print('iThreadIndex_x', iThreadIndex_x)
    #     print('iBatchIndex', iBatchIndex)
    #     print('iBatchIndex_rem', iBatchIndex_rem)
    #     print('iPolIndex', iPolIndex)
    #     print('iPolIndex_rem', iPolIndex_rem)
    #     print('iChannelIndex:', iChannelIndex)
    #     print('iChannelIndex_rem:', iChannelIndex_rem)
    #     print('iAntIndex:', iAntIndex)
    #     print('iAntIndex_rem:', iAntIndex_rem)
    #     print('iBeamIndex:', iBeamIndex)
    #
    #     print('iAntMatrix', iAntMatrix)
    #     print('iBeamMatrix', iBeamMatrix)

    # coeffs[iBeamIndex][iChannelIndex][iAntIndex][0] = SteeringCoeffCorrectReal
    # coeffs[iBeamIndex][iChannelIndex][iAntIndex][1] = SteeringCoeffCorrectImag
    # coeffs[iBeamIndex][iChannelIndex][iAntIndex][0] = -SteeringCoeffCorrectImag
    # coeffs[iBeamIndex][iChannelIndex][iAntIndex][1] = SteeringCoeffCorrectReal

    # coeffs[iBeamIndex][iChannelIndex][iAntIndex][0][0] = SteeringCoeffCorrectReal
    # coeffs[iBeamIndex][iChannelIndex][iAntIndex][0][1] = SteeringCoeffCorrectImag
    # coeffs[iBeamIndex][iChannelIndex][iAntIndex][1][0] = -SteeringCoeffCorrectImag
    # coeffs[iBeamIndex][iChannelIndex][iAntIndex][1][1] = SteeringCoeffCorrectReal

    # if (iBeamMatrix%2==1):
    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix][iBeamMatrix+1] = SteeringCoeffCorrectImag #1
    # else:
    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix][iBeamMatrix] = SteeringCoeffCorrectReal #4

    # if (iAntMatrix%2==1):
    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix+1][iBeamMatrix+1] = SteeringCoeffCorrectReal #4
    # else:
    coeffs[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix+1][iBeamMatrix] = -SteeringCoeffCorrectImag #-1

    # if iThreadIndex_x == 4095:
        # if (iBeamMatrix % 2 == 1):
        #     print('Odd: iBeamMatrix', iBeamMatrix)
        # coeffs[iChannelIndex][iAntMatrix][iBeamMatrix+1] = 1
        # # else:
        # #     print('Even: iBeamMatrix', iBeamMatrix)
        # coeffs[iChannelIndex][iAntMatrix][iBeamMatrix] = 4
        #
        # # if (iAntMatrix % 2 == 1):
        # #     print('Odd: iAntMatrix', iAntMatrix)
        # coeffs[iChannelIndex][iAntMatrix][iBeamMatrix+1] = -4
        # # else:
        # #     print('Even: iAntMatrix', iAntMatrix)
        # coeffs[iChannelIndex][iAntMatrix][iBeamMatrix] = 1

    # iInterChannelIndex = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # if iInterChannelIndex < 1:
    #     print('NBEAMS x ANTS:', n_beams*n_ants)
    #     print(n_beams*n_ants)
    #     print(iInterChannelIndex)

    # if (iInterChannelIndex < n_beams*n_ants):
    #     iChannelIndex = cuda.blockIdx.y
    #     iAntIndex = np.uint(iInterChannelIndex/n_beams)
    #     iBeamIndex = iInterChannelIndex - iAntIndex * n_beams
    #
    #     fDelay_s = delay_vals[0][iAntIndex*n_beams + iBeamIndex]
    #     fDelayRate_sps = delay_vals[1][iAntIndex*n_beams + iBeamIndex]
    #     fPhase_rad = delay_vals[2][iAntIndex*n_beams + iBeamIndex]
    #     fPhaseRate_radps = delay_vals[3][iAntIndex*n_beams + iBeamIndex]
    #
    #     if iInterChannelIndex == 0:
    #         print('iAntIndex*n_beams + iBeamIndex:', iAntIndex*n_beams + iBeamIndex)
    #         print(fDelay_s)
    #         print(fDelayRate_sps)
    #         print(fPhase_rad)
    #         print(fPhaseRate_radps)
    #
    #     fTimeDifference = current_time_sec - ref_time_sec
    #     fNanosecondsTimeDifference = current_time_ns - ref_time_ns
    #     fTimeDifference += fNanosecondsTimeDifference/1e9
    #
    #     fDeltaTime = fTimeDifference
    #     fDeltaDelay = fDelayRate_sps*fDeltaTime
    #     fDelayN = (fDelayRate_sps + fDeltaDelay) * iChannelIndex * (np.pi) / (
    #                 SAMPLING_PERIOD * n_channels)
    #     fDelayN2 = (fDelay_s + fDeltaDelay)*(n_channels/2)*(np.pi)/(SAMPLING_PERIOD*n_channels)
    #     fDeltaPhase = fPhaseRate_radps*fDeltaTime
    #     fPhase0 = fPhase_rad - fDelayN2 + fDeltaPhase
    #     fRotation = fDelayN + fPhase0
    #
    #     fSteeringCoeffCorrectReal = math.cos(fRotation)
    #     fSteeringCoeffCorrectImag = math.sin(fRotation)
    #     if iInterChannelIndex == 0:
    #         print('Real:', fSteeringCoeffCorrectReal)
    #         print('Imag:', fSteeringCoeffCorrectImag)
    #
    #     ulOutputIndex = 0
    #     coeffs[ulOutputIndex] = fSteeringCoeffCorrectReal
    #     coeffs[ulOutputIndex+1] = fSteeringCoeffCorrectImag

class beamform_coeff_kernel:
    """Class for beamform complex multiplication."""

    def coeff_gen(current_time, ref_time, delay_vals, batches, pols, n_beams, num_channels, n_ants):
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
        # n_channel = coeff_matrix.shape[0]
        # n_beams = coeff_matrix.shape[1]
        # n_ants = coeff_matrix.shape[2]
        complexity = 2
        cols = 2

        # fDelay_s = delay_vals[0]
        # fDelayRate_sps = delay_vals[1]
        # fPhase_rad = delay_vals[2]
        # fPhaseRate_radps = delay_vals[3]

        # Temp
        coeff_matrix = np.empty(batches*pols*num_channels*n_ants*n_beams*complexity*cols, dtype=np.float32)
        coeff_matrix = coeff_matrix.reshape(batches, pols, num_channels, n_ants*complexity, n_beams*cols)

        # Set the number of threads in a block
        threadsperblock = 128

        # Calculate the number of thread blocks in the grid
        blockspergrid = np.uint((batches*pols*num_channels*n_beams*n_ants) // threadsperblock)

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue. This will make the
        # context associated with the deivce device_id the current context.
        cuda.select_device(0)

        # delay_vals = delay_vals.reshape(n_beams*num_channels*n_ants*4)
        # Now start the kernel
        # run_coeff_gen[blockspergrid, threadsperblock](current_time, ref_time, fDelay_s, fDelayRate_sps, fPhase_rad, fPhaseRate_radps, n_channel, n_beams, n_ants, coeff_matrix)
        run_coeff_gen[blockspergrid, threadsperblock](current_time, ref_time, delay_vals, batches, pols, num_channels, n_beams, n_ants, coeff_matrix)

        # Wait for all commands in the stream to finish executing.
        cuda.synchronize()

        return coeff_matrix
