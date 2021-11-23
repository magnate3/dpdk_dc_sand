"""
Module for beamformer complex multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
from numba import cuda, float32


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
    debug_thread_idx = 4095
    # # Compute flattened index inside the array
    iThreadIndex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    pols = data_matrix.shape[1]
    n_channel = data_matrix.shape[2]
    blocks = data_matrix.shape[3]
    samples_per_block = data_matrix.shape[4]
    ants = data_matrix.shape[5]
    n_beams = 8

    # Compute data matrix index
    iBatchIndex = iThreadIndex_x // (pols * n_channel * blocks * samples_per_block)
    iRemIndex = iThreadIndex_x % (pols * n_channel * blocks * samples_per_block)
    if iThreadIndex_x == debug_thread_idx:
        print(' ')
        print('iThreadIndex_x', iThreadIndex_x)
        print('iBatchIndex:', iBatchIndex)
        print('iRemIndex:', iRemIndex)

    iPolIndex = iRemIndex // (n_channel * blocks * samples_per_block)
    iRemIndex = iRemIndex % (n_channel * blocks * samples_per_block)
    if iThreadIndex_x == debug_thread_idx:
        print('iPolIndex', iPolIndex)
        print('iRemIndex', iRemIndex)

    iChanIndex = iRemIndex // (blocks * samples_per_block)
    iRemIndex = iRemIndex % (blocks * samples_per_block)

    if iThreadIndex_x == debug_thread_idx:
        print('iChanIndex', iChanIndex)
        print('iRemIndex', iRemIndex)

    iBlockIndex = iRemIndex // (samples_per_block)
    iRemIndex = iRemIndex % (samples_per_block)
    iSamplePerBlockIndex = iRemIndex

    if iThreadIndex_x == debug_thread_idx:
        print('iBlockIndex', iBlockIndex)
        print('iRemIndex', iRemIndex)
        print('iSamplePerBlockIndex', iSamplePerBlockIndex)

    # Compute Coeff matrix index
    icBatchIndex = iThreadIndex_x // (pols*n_channel*n_beams*2*ants*2)
    icBatchIndex_rem = iThreadIndex_x % (pols*n_channel*n_beams*2*ants*2)

    icPolIndex = icBatchIndex_rem // (n_channel*n_beams*2 * ants*2)
    icPolIndex_rem = icBatchIndex_rem % (n_channel*n_beams*2 * ants*2)

    icChannelIndex = icPolIndex_rem // (n_beams*2*ants*2)
    icChannelIndex_rem = icPolIndex_rem % (n_beams*2*ants*2)

    icAntIndex = icChannelIndex_rem // (n_beams*2*2)
    icAntIndex_rem = icChannelIndex_rem % (n_beams*2*2)

    icAntMatrix = icAntIndex*2

    icBeamIndex = icAntIndex_rem//(2*2)
    icBeamMatrix = icBeamIndex*2

    if iThreadIndex_x == debug_thread_idx:
        print(' ')
        print(' ')
        print('Coeff Index:')
        print('iThreadIndex_x', iThreadIndex_x)
        print('icBatchIndex', icBatchIndex)
        print('icBatchIndex_rem', icBatchIndex_rem)
        print('icPolIndex', icPolIndex)
        print('icPolIndex_rem', icPolIndex_rem)
        print('icChannelIndex:', icChannelIndex)
        print('icChannelIndex_rem:', icChannelIndex_rem)
        print('icAntIndex:', icAntIndex)
        print('icAntIndex_rem:', icAntIndex_rem)
        print('icBeamIndex:', icBeamIndex)
        print('icAntMatrix', icAntMatrix)
        print('icBeamMatrix', icBeamMatrix)
        # coeff = coeff_matrix[iBatchIndex][iPolIndex][iChannelIndex][iAntMatrix][iBeamMatrix]
        # print('Coeff:', coeff)

    # for col in range(2):
    for col in range(n_beams*2):
        tmp = float32(0)
        for ant in range(ants):
            # coeff = coeff_matrix[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][col][ant]
            coeff = coeff_matrix[icBatchIndex][icPolIndex][icChannelIndex][ant][col]
            data = data_matrix[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][ant]

            tmp += data * coeff

            if (iThreadIndex_x == debug_thread_idx) & (col==14):
                # print('BF icChannelIndex:', icChannelIndex)
                # print('-----')
                print('BF Col:', col)
                print('BF Ant:', ant)
                print('data', data)
                print('coeff', coeff)
                print('tmp', tmp)

        # Copy computed weighted and summed ant samples to output
        # if col%2 == 0:
            # Computed sample is real component
        out[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][col] = tmp

        if (iThreadIndex_x == debug_thread_idx):
            print('out:', tmp)
        # else:
            # Computed sample is imaginary component
            # out[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][col] = tmp

        # Copy computed weighted and summed ant samples to output
        # if col == 0:
            # Computed sample is real component
            # out[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][0] = tmp
        # else:
            # Computed sample is imaginary component
            # out[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][1] = tmp


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

        # Reshape data to have ant real and imag data in one dimension
        data_matrix = data_matrix.reshape(batches, pols, n_channel, blocks, samples_per_block, (ants * complexity))

        # Set the number of threads in a block
        threadsperblock = 512

        # Calculate the number of thread blocks in the grid
        ant_sample_blocks = data_matrix.size / (ants * complexity)
        blockspergrid = int(ant_sample_blocks // threadsperblock)

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue. This will make the
        # context associated with the deivce device_id the current context.
        cuda.select_device(0)

        # Now start the kernel
        run_complex_mult[blockspergrid, threadsperblock](data_matrix, coeff_matrix, out)

        # Wait for all commands in the stream to finish executing.
        cuda.synchronize()
