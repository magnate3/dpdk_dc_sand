from __future__ import division
import numpy as np
from numba import cuda

# def run_complex_mult(self, data_matrix, coeff_matrix, out=None):
@cuda.jit
def run_complex_mult(io_array):
    """
    Code for kernel.
    """

    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    #print(tx)
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < io_array.size:  # Check array boundaries
        io_array[pos] *= 2 # do the computation
    # code here

class complex_mult_kernel:
    def complex_mult(self, data_matrix, coeff_matrix, out=None, stream=None):
        batches = data_matrix.shape[0]
        pols = data_matrix.shape[1]
        n_channel = data_matrix.shape[2]
        blocks = data_matrix.shape[3] 
        samples_per_block = data_matrix.shape[4]
        ants = data_matrix.shape[5]
        complexity = 2 # always

        # Create the data array - usually initialized some other way
        data = np.ones(256)

        # Set the number of threads in a block
        threadsperblock = 32 

        # Calculate the number of thread blocks in the grid
        blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue.
        cuda.select_device(0)

        # Now start the kernel
        run_complex_mult[blockspergrid, threadsperblock](data)

        # Print the result
        print(data)
