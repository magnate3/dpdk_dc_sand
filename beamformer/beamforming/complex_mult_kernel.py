"""
Module for beamformer complex multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
import numpy as np
from numba import cuda, float32


@cuda.jit
def run_complex_mult(data_matrix: np.ndarray, coeff_matrix: np.ndarray, out: np.ndarray):
    """Execute complex multiplication.

    Parameters
    ----------
    data_matrix:
        Data matrix on reordered data
    coeff_matrix:
        Coefficients for beamforming computation.
    out:
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
    # Compute flattened index inside the array
    ithreadindex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    batches = data_matrix.shape[0]
    pols = data_matrix.shape[1]
    n_channel = data_matrix.shape[2]
    blocks = data_matrix.shape[3]
    samples_per_block = data_matrix.shape[4]
    ants = data_matrix.shape[5]
    complexity = 2
    n_beams = coeff_matrix.shape[4] // complexity


    if ithreadindex_x <= 0:
        print(  'thread:',ithreadindex_x,
                'batches:', batches,
                'pols:', pols,
                'n_channel:', n_channel,
                'blocks:', blocks,
                'samples_per_block:', samples_per_block,
                'ants:', ants,
                'complexity:', complexity, 
                'n_beams:', n_beams)
    

    if ithreadindex_x <= (batches * pols * n_channel * blocks * samples_per_block * ants * complexity):
        # Compute data matrix index
        ibatchindex = ithreadindex_x // (pols * n_channel * blocks * samples_per_block * ants)
        iremindex = ithreadindex_x % (pols * n_channel * blocks * samples_per_block * ants)

        ipolindex = iremindex // (n_channel * blocks * samples_per_block * ants)
        iremindex = iremindex % (n_channel * blocks * samples_per_block * ants)

        ichanindex = iremindex // (blocks * samples_per_block * ants)
        iremindex = iremindex % (blocks * samples_per_block * ants)

        iblockindex = iremindex // (samples_per_block * ants)
        iremindex = iremindex % (samples_per_block * ants)

        isample_per_block_index = iremindex // ants

        if ithreadindex_x == 0:
            print('ants is:', ants)

        for col in range(n_beams * 2):
            tmp = float32(0)
            for ant in range(ants):
                coeff = coeff_matrix[ibatchindex][ipolindex][ichanindex][ant][col]
                data = data_matrix[ibatchindex][ipolindex][ichanindex][iblockindex][isample_per_block_index][ant]
                tmp += data * coeff

                if ithreadindex_x <= 0:
                    print('thread is:',ithreadindex_x,
                    'ibatchindex:', ibatchindex,
                    'ipolindex:', ipolindex,
                    'ichanindex:', ichanindex,
                    'iblockindex:', iblockindex,
                    'isample_per_block_index:', isample_per_block_index,
                    'ant:', ant,
                    'col:', col, 
                    'tmp:', tmp, 
                    'coeff:',coeff, 
                    'data:', data)

                if ibatchindex == 0:
                    if ipolindex == 0:
                        if ichanindex == 0:
                            if iblockindex == 0:
                                if isample_per_block_index == 0:
                                    if ant == 0:
                                        a = 1
                                        # print(ithreadindex_x)
                                        # print('col is:', col, 'and tmp is:', tmp, 'and coeff is:',coeff, 'and data is:', data)

            out[ibatchindex][ipolindex][ichanindex][iblockindex][isample_per_block_index][col] = tmp


class ComplexMultKernel:
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

        # Calculate the number of threads required.
        total_threads = batches * pols * n_channel * blocks * samples_per_block * ants * complexity

        threadsperblock = 128

        # Calculate the number of blocks in a grid.
        blockspergrid = np.uint(np.ceil(total_threads / threadsperblock))

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue. This will make the
        # context associated with the deivce device_id the current context.
        cuda.select_device(0)

        # Now start the kernel
        run_complex_mult[blockspergrid, threadsperblock](data_matrix, coeff_matrix, out)

        # Wait for all commands in the stream to finish executing.
        cuda.synchronize()
