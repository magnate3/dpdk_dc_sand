"""
Module for beamformer complex multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
from numba import cuda, float32
import math
import numpy as np

@cuda.jit
def run_coeff_gen(current_time_sec, ref_time_sec, delay_vals, n_channels, n_beams, n_ants, coeffs):
    # def run_coeff_gen(current_time, ref_time, delay, n_channel, n_beams, n_ants, coeff_matrix):

    """Execute complex multiplication.
    """
    # # Compute flattened index inside the array

    current_time_ns = current_time_sec
    ref_time_ns = ref_time_sec
    SAMPLING_PERIOD = 1e-7

    iInterChannelIndex = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if iInterChannelIndex < 1:
        print('NBEAMS x ANTS:', n_beams*n_ants)
        print(n_beams*n_ants)
        print(iInterChannelIndex)

    if (iInterChannelIndex < n_beams*n_ants):
        iChannelIndex = cuda.blockIdx.y
        iAntIndex = np.uint(iInterChannelIndex/n_beams)
        iBeamIndex = iInterChannelIndex - iAntIndex * n_beams

        fDelay_s = delay_vals[0][iAntIndex*n_beams + iBeamIndex]
        fDelayRate_sps = delay_vals[1][iAntIndex*n_beams + iBeamIndex]
        fPhase_rad = delay_vals[2][iAntIndex*n_beams + iBeamIndex]
        fPhaseRate_radps = delay_vals[3][iAntIndex*n_beams + iBeamIndex]

        if iInterChannelIndex == 0:
            print('iAntIndex*n_beams + iBeamIndex:', iAntIndex*n_beams + iBeamIndex)
            print(fDelay_s)
            print(fDelayRate_sps)
            print(fPhase_rad)
            print(fPhaseRate_radps)

        fTimeDifference = current_time_sec - ref_time_sec
        fNanosecondsTimeDifference = current_time_ns - ref_time_ns
        fTimeDifference += fNanosecondsTimeDifference/1e9

        fDeltaTime = fTimeDifference
        fDeltaDelay = fDelayRate_sps*fDeltaTime
        fDelayN = (fDelayRate_sps + fDeltaDelay) * iChannelIndex * (np.pi) / (
                    SAMPLING_PERIOD * n_channels)
        fDelayN2 = (fDelay_s + fDeltaDelay)*(n_channels/2)*(np.pi)/(SAMPLING_PERIOD*n_channels)
        fDeltaPhase = fPhaseRate_radps*fDeltaTime
        fPhase0 = fPhase_rad - fDelayN2 + fDeltaPhase
        fRotation = fDelayN + fPhase0

        fSteeringCoeffCorrectReal = math.cos(fRotation)
        fSteeringCoeffCorrectImag = math.sin(fRotation)
        if iInterChannelIndex == 0:
            print('Real:', fSteeringCoeffCorrectReal)
            print('Imag:', fSteeringCoeffCorrectImag)

        ulOutputIndex = 0
        coeffs[ulOutputIndex] = fSteeringCoeffCorrectReal
        coeffs[ulOutputIndex+1] = fSteeringCoeffCorrectImag

class beamform_coeff_kernel:
    """Class for beamform complex multiplication."""

    def coeff_gen(current_time, ref_time, delay_vals):
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
        n_channels = 128
        n_beams = 16
        n_ants = 64
        complexity = 2
        beam_ant_pair = n_beams * n_ants

        # Temp
        coeff_matrix = np.empty(n_channels*n_ants*complexity*n_beams*2, dtype=np.float)

        coeff_matrix = coeff_matrix.reshape(n_beams,n_channels,n_ants*complexity,2)

        fDelay_s = delay_vals[0]
        fDelayRate_sps = delay_vals[1]
        fPhase_rad = delay_vals[2]
        fPhaseRate_radps = delay_vals[3]

        # Set the number of threads in a block
        threadsperblock = 128

        # Calculate the number of thread blocks in the grid
        blockspergrid = int(beam_ant_pair // threadsperblock)

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue. This will make the
        # context associated with the deivce device_id the current context.
        cuda.select_device(0)

        # Now start the kernel
        # run_coeff_gen[blockspergrid, threadsperblock](current_time, ref_time, fDelay_s, fDelayRate_sps, fPhase_rad, fPhaseRate_radps, n_channel, n_beams, n_ants, coeff_matrix)
        run_coeff_gen[blockspergrid, threadsperblock](current_time, ref_time, delay_vals, n_channels, n_beams, n_ants, coeff_matrix)

    # Wait for all commands in the stream to finish executing.
        cuda.synchronize()
