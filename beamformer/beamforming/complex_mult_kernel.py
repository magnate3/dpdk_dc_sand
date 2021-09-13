"""
Module for beamformer complex multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
from numba import cuda, float32

# from numba import jit


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
    # # Compute flattened index inside the array
    iThreadIndex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    pols = data_matrix.shape[1]
    n_channel = data_matrix.shape[2]
    blocks = data_matrix.shape[3]
    samples_per_block = data_matrix.shape[4]
    ants = data_matrix.shape[5]

    iBatchIndex = iThreadIndex_x // (pols * n_channel * blocks * samples_per_block)
    iRemIndex = iThreadIndex_x % (pols * n_channel * blocks * samples_per_block)

    iPolIndex = iRemIndex // (n_channel * blocks * samples_per_block)
    iRemIndex = iRemIndex % (n_channel * blocks * samples_per_block)

    iChanIndex = iRemIndex // (blocks * samples_per_block)
    iRemIndex = iRemIndex % (blocks * samples_per_block)

    iBlockIndex = iRemIndex // (samples_per_block)
    iRemIndex = iRemIndex % (samples_per_block)

    iSamplePerBlockIndex = iRemIndex

    for col in range(2):
        tmp = float32(0)
        for ant in range(ants):
            coeff = coeff_matrix[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][col][ant]
            data = data_matrix[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][ant]

            tmp += data * coeff

        # Copy computed weighted and summed ant samples to output
        if col == 0:
            # Computed sample is real component
            out[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][0] = tmp
        else:
            # Computed sample is imaginary component
            out[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][1] = tmp


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
        # the device already has a context associated to it from katsdpsigproc command queue.
        cuda.select_device(0)

        # Now start the kernel
        run_complex_mult[blockspergrid, threadsperblock](data_matrix, coeff_matrix, out)
