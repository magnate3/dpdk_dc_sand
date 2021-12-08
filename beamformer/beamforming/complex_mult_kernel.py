"""
Module for beamformer complex multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
from numba import cuda, float32
import numpy as np

@cuda.jit
def run_complex_mult(data_matrix, coeff_matrix, out):
    """Execute complex multiplication.

    Parameters
    ----------
    data_matrix: nd.array[np.uint8]
        Data matrix on reordered data
    coeff_matrix: nd.array[np.float32]
        Coefficients for beamforming computation.
    out: nd.array[np.float32]
        Complex multipication product for beamforming computation.

    Note: This is for use in complex multiplication using two
    real-valued arrays. For this reason the coefficients need to be
    arranged as follows.

    Coefficients Array:
    [R00  I00
     -I00 R00
     R10  I10
     -I10 R10
     ...  ...]
    Where:  R00 = Real coeff 0
            I00 = Imag coeff 0
    and the matrix is structured as a N x 2 array.

    Each GPU thread executes on one sample set of all antennas. The set of antenna
    samples are multiplied with the coefficeint matrix and accumulated. Both the
    data and the coefficients used are complex valued requiring a complex multiplication.
    To utilise standard matrix mutliplication the coefficient matrix is constructed as detailed above.
    """
    # debug_thread_idx_lower = 2*(2*(32*16*16*4*2)) + 1*(32*16*16*4*2) + (34*16*16*8*2)
    debug_thread_idx_lower = 0
    debug_thread_idx_upper = debug_thread_idx_lower + 16

    # # Compute flattened index inside the array
    iThreadIndex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # if (iThreadIndex_x > debug_thread_idx_lower)&(iThreadIndex_x < debug_thread_idx_upper):
    #     print('debug_thread_idx_lower', debug_thread_idx_lower)
    #     print('debug_thread_idx_upper', debug_thread_idx_upper)

    batches = data_matrix.shape[0]
    pols = data_matrix.shape[1]
    n_channel = data_matrix.shape[2]
    blocks = data_matrix.shape[3]
    samples_per_block = data_matrix.shape[4]
    ants = data_matrix.shape[5]
    n_beams = 8

    if (iThreadIndex_x == debug_thread_idx_lower):
        print('ants', ants)
    # Compute data matrix index
    iBatchIndex = iThreadIndex_x // (pols * n_channel * blocks * samples_per_block * ants)
    iRemIndex = iThreadIndex_x % (pols * n_channel * blocks * samples_per_block * ants)
    # if (iThreadIndex_x > debug_thread_idx_lower)&(iThreadIndex_x < debug_thread_idx_upper):
    #     print(' ')
    #     print('iThreadIndex_x', iThreadIndex_x)
    #     print('iBatchIndex:', iBatchIndex)
    #     print('iRemIndex:', iRemIndex)

    iPolIndex = iRemIndex // (n_channel * blocks * samples_per_block * ants)
    iRemIndex = iRemIndex % (n_channel * blocks * samples_per_block * ants)
    # if (iThreadIndex_x > debug_thread_idx_lower)&(iThreadIndex_x < debug_thread_idx_upper):
    #     print('iPolIndex', iPolIndex)
    #     print('iRemIndex', iRemIndex)

    iChanIndex = iRemIndex // (blocks * samples_per_block * ants)
    iRemIndex = iRemIndex % (blocks * samples_per_block * ants)

    # if (iThreadIndex_x > debug_thread_idx_lower)&(iThreadIndex_x < debug_thread_idx_upper):
    #     print('iChanIndex', iChanIndex)
    #     print('iRemIndex', iRemIndex)

    iBlockIndex = iRemIndex // (samples_per_block * ants)
    iRemIndex = iRemIndex % (samples_per_block * ants)

    # if (iThreadIndex_x > debug_thread_idx_lower)&(iThreadIndex_x < debug_thread_idx_upper):
    #     print('iBlockIndex', iBlockIndex)
    #     print('iRemIndex', iRemIndex)

    iSamplePerBlockIndex = iRemIndex // ants
    # iRemIndex = iRemIndex % ants

    # IAntIndex = iRemIndex

    # if (iThreadIndex_x > debug_thread_idx_lower)&(iThreadIndex_x < debug_thread_idx_upper):
    #     print('iSamplePerBlockIndex', iSamplePerBlockIndex)
    #     print('IAntIndex', IAntIndex)

    # Compute Coeff matrix index
    # icBatchIndex = iThreadIndex_x // (pols*n_channel * ants * n_beams*2)
    # icBatchIndex_rem = iThreadIndex_x % (pols*n_channel * ants * n_beams*2)
    #
    # icPolIndex = icBatchIndex_rem // (n_channel * ants * n_beams*2)
    # icPolIndex_rem = icBatchIndex_rem % (n_channel * ants * n_beams*2)
    #
    # icChannelIndex = icPolIndex_rem // (samples_per_block * ants * n_beams*2)
    # icChannelIndex_rem = icPolIndex_rem % (samples_per_block * ants * n_beams*2)

    # icAntIndex = icChannelIndex_rem // (n_beams*2*2)
    # icAntIndex_rem = icChannelIndex_rem % (n_beams*2*2)
    #
    # icAntMatrix = icAntIndex*2
    #
    # icBeamIndex = icAntIndex_rem//(2*2)
    # icBeamMatrix = icBeamIndex*2

    # icChannelIndex = icPolIndex_rem // (ants * n_beams*2)
    # icChannelIndex_rem = icPolIndex_rem % (ants * n_beams*2)
    #
    # icAntIndex = icChannelIndex_rem // (n_beams*2*2)
    # icAntIndex_rem = icChannelIndex_rem % (n_beams*2*2)
    #
    # icAntMatrix = icAntIndex*2
    #
    # icBeamIndex = icAntIndex_rem//(2*2)
    # icBeamMatrix = icBeamIndex*2

    # if (iThreadIndex_x > debug_thread_idx_lower)&(iThreadIndex_x < debug_thread_idx_upper):
    #     print(' ')
    #     print(' ')
    #     print('icBatchIndex', icBatchIndex)
    #     print('icBatchIndex_rem', icBatchIndex_rem)
    #     print('icPolIndex', icPolIndex)
    #     print('icPolIndex_rem', icPolIndex_rem)
    #     print('icChannelIndex:', icChannelIndex)
    #     print('icChannelIndex_rem:', icChannelIndex_rem)
        # print('icAntIndex:', icAntIndex)
        # print('icAntIndex_rem:', icAntIndex_rem)
        # print('icBeamIndex:', icBeamIndex)
        # print('icAntMatrix', icAntMatrix)
        # print('icBeamMatrix', icBeamMatrix)
        # coeff = coeff_matrix[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix][iBeamMatrix]
        # print('Coeff:', coeff)

    for col in range(n_beams*2):
        tmp = float32(0)
        for ant in range(ants):
            coeff = coeff_matrix[iBatchIndex][iPolIndex][iChanIndex][ant][col]
            # coeff = 0
            data = data_matrix[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][ant]
            # data = 0
            tmp += data * coeff

            # if (iThreadIndex_x > debug_thread_idx_lower)&(iThreadIndex_x < debug_thread_idx_upper):
            #     print('thread:', iThreadIndex_x,
            #           'iBatchIndex', iBatchIndex,
            #           'icBatchIndex',icBatchIndex,
            #           'iPolIndex:', iPolIndex,
            #           'icPolIndex:', icPolIndex,
            #           'iChanIndex:', iChanIndex,
            #           'icChannelIndex:', icChannelIndex,
            #           'iBlockIndex:', iBlockIndex,
            #           'iSamplePerBlockIndex:', iSamplePerBlockIndex,
            #           'ant:', ant,
            #           'col:', col,
            #           'coeff:', coeff,
            #           'data:', data,
            #           'tmp', tmp)



        # Copy computed weighted and summed ant samples to output
        # if col%2 == 0:
            # Computed sample is real component

        # if (iThreadIndex_x < debug_thread_idx)&(col>=0):
        #     print('thread:', iThreadIndex_x, 'ant:', ant, 'col:', col, 'coeff:', coeff, 'data:', data, 'tmp', tmp)

        # if (iThreadIndex_x > debug_thread_idx_lower)&(iThreadIndex_x < debug_thread_idx_upper):
        #     print('thread:', iThreadIndex_x,
        #         'iBatchIndex', iBatchIndex,
        #         'iPolIndex:', iPolIndex,
        #         'iChanIndex:', iChanIndex,
        #         'iBlockIndex:', iBlockIndex,
        #         'iSamplePerBlockIndex:', iSamplePerBlockIndex,
        #         'ant:', ant,
        #         'col:', col,
        #         'coeff:', coeff,
        #         'data:', data,
        #         'tmp', tmp)

        out[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][col] = tmp

        # if iBatchIndex > batches:
        #     print('iThreadIndex_x', iThreadIndex_x, 'iBatchIndex', iBatchIndex)


class complex_mult_kernel:
    """Class for beamform complex multiplication."""

    def complex_mult(self, data_matrix, coeff_matrix, out):
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
        batches = data_matrix.shape[0]
        pols = data_matrix.shape[1]
        n_channel = data_matrix.shape[2]
        blocks = data_matrix.shape[3]
        samples_per_block = data_matrix.shape[4]
        ants = data_matrix.shape[5]
        complexity = 2  # always
        n_beams = coeff_matrix.shape[4]//complexity


        # Reshape data to have ant real and imag data in one dimension
        data_matrix = data_matrix.reshape(batches, pols, n_channel, blocks, samples_per_block, (ants * complexity))

        # Set the number of threads in a block
        # threadsperblock = 512

        # Calculate the number of thread blocks in the grid
        # ant_sample_blocks = data_matrix.size / (ants * complexity)
        # ant_sample_blocks = data_matrix.size
        # ant_sample_blocks = batches * pols * n_channel * blocks * samples_per_block * n_beams * complexity
        ant_sample_blocks = batches * pols * n_channel * blocks * samples_per_block * ants * complexity

        largest_divisor = 0
        num = ant_sample_blocks
        for i in range(2, num):
            if ((num % i == 0) & (i<=1024)):
                largest_divisor = i
            elif (i > 1024):
                break

        threadsperblock = largest_divisor

        blockspergrid = np.uint(ant_sample_blocks // threadsperblock)
        # blockspergrid = np.uint(np.ceil(ant_sample_blocks / threadsperblock))

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue. This will make the
        # context associated with the deivce device_id the current context.
        cuda.select_device(0)

        # Now start the kernel
        run_complex_mult[blockspergrid, threadsperblock](data_matrix, coeff_matrix, out)

        # Wait for all commands in the stream to finish executing.
        cuda.synchronize()
