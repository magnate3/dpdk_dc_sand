from __future__ import division
import numpy as np
from numba import cuda, float32

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

# def run_complex_mult(self, data_matrix, coeff_matrix, out=None):
@cuda.jit
def run_complex_mult(data_matrix, coeff_matrix, out):
    """
    Code for kernel.
    """
    # # Compute flattened index inside the array
    iThreadIndex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    batches = data_matrix.shape[0]
    pols = data_matrix.shape[1]
    n_channel = data_matrix.shape[2]
    blocks = data_matrix.shape[3] 
    samples_per_block = data_matrix.shape[4]
    ants = data_matrix.shape[5]
    complexity = 2 # always

    # iBatchCounter = cuda.blockIdx.y

    # iMatrixStride_y = iBatchCounter * pols * n_channel * blocks * samples_per_block * ants

    iBatchIndex = int(iThreadIndex_x / (pols * n_channel * blocks * samples_per_block))
    iRemIndex = iThreadIndex_x % (pols * n_channel * blocks * samples_per_block)

    if iThreadIndex_x >= 98303:
        print('iThreadIndex_x is:', iThreadIndex_x)
        print('iBatchIndex is:',iBatchIndex)
        print('iRemIndex is:',iRemIndex)

    iPolIndex = int(iRemIndex / (n_channel * blocks * samples_per_block))
    iRemIndex = iRemIndex % (n_channel * blocks * samples_per_block)

    if iThreadIndex_x >= 98303:
        print('iPolIndex is:', iPolIndex)
        print('iRemIndex is:',iRemIndex)

    iChanIndex = int(iRemIndex / (blocks * samples_per_block))
    iRemIndex = iRemIndex % (blocks * samples_per_block)

    if iThreadIndex_x >= 98303:
        print('iChanIndex is:', iChanIndex)
        print('iRemIndex is:',iRemIndex)

    iBlockIndex = int(iRemIndex / (samples_per_block))
    iRemIndex = iRemIndex % (samples_per_block)

    if iThreadIndex_x >= 98303:
        print('iBlockIndex is:', iBlockIndex)
        print('iRemIndex is:',iRemIndex)

    iSamplePerBlockIndex = iRemIndex

    if iThreadIndex_x >= 98303:
        print('iSamplePerBlockIndex is:', iSamplePerBlockIndex)

    tmp = data_matrix[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex]

    if iThreadIndex_x >= 98303:
        for ant in range(ants):
            print(ant)
            print(tmp[ant][0])
            print(tmp[ant][1])


    # # Define an array in the shared memory
    # # The size and type of the arrays must be known at compile time
    # sdata_matrix = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    # scoeff_matrix = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    # x, y = cuda.grid(2)
    
    # tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y

    # tmp = 0.
    # for i in range(int(data_matrix.shape[1] / TPB)):
    #     # Preload data into shared memory
    #     sdata_matrix[tx, ty] = data_matrix[x, ty + i * TPB]
    #     scoeff_matrix[tx, ty] = coeff_matrix[tx + i * TPB, y]

    #     # Wait until all threads finish preloading
    #     cuda.syncthreads()





    # if pos < io_array.size:  # Check array boundaries
    #     io_array[pos] *= 2 # do the computation
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
        # data = np.ones(256)

        # Set the number of threads in a block
        threadsperblock = 512 

        # Calculate the number of thread blocks in the grid
        ant_sample_blocks = data_matrix.size / (ants*complexity)
        blockspergrid = int(ant_sample_blocks // threadsperblock)

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue.
        cuda.select_device(0)

        # Now start the kernel
        run_complex_mult[blockspergrid, threadsperblock](data_matrix, coeff_matrix, out)

        a = 2
        # Print the result
        # print(data)
