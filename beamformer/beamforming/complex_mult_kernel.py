from __future__ import division
import numpy as np
from numba import cuda, float32

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
# TPB = 16

@cuda.jit
def run_complex_mult(data_matrix, coeff_matrix, out):
    """
    Code for kernel.
    """
    dbg_trig = 98303 * 4

    # # Compute flattened index inside the array
    iThreadIndex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    batches = data_matrix.shape[0]
    pols = data_matrix.shape[1]
    n_channel = data_matrix.shape[2]
    blocks = data_matrix.shape[3] 
    samples_per_block = data_matrix.shape[4]
    ants = data_matrix.shape[5]
    complexity = 2 # always

    iBatchIndex = int(iThreadIndex_x / (pols * n_channel * blocks * samples_per_block))
    iRemIndex = iThreadIndex_x % (pols * n_channel * blocks * samples_per_block)

    # if iThreadIndex_x >= dbg_trig:
    #     print('iThreadIndex_x is:', iThreadIndex_x)
    #     print('iBatchIndex is:',iBatchIndex)
    #     print('iRemIndex is:',iRemIndex)

    iPolIndex = int(iRemIndex / (n_channel * blocks * samples_per_block))
    iRemIndex = iRemIndex % (n_channel * blocks * samples_per_block)

    # if iThreadIndex_x >= dbg_trig:
    #     print('iPolIndex is:', iPolIndex)
    #     print('iRemIndex is:',iRemIndex)

    iChanIndex = int(iRemIndex / (blocks * samples_per_block))
    iRemIndex = iRemIndex % (blocks * samples_per_block)

    # if iThreadIndex_x >= dbg_trig:
    #     print('iChanIndex is:', iChanIndex)
    #     print('iRemIndex is:',iRemIndex)

    iBlockIndex = int(iRemIndex / (samples_per_block))
    iRemIndex = iRemIndex % (samples_per_block)

    # if iThreadIndex_x >= dbg_trig:
    #     print('iBlockIndex is:', iBlockIndex)
    #     print('iRemIndex is:',iRemIndex)

    iSamplePerBlockIndex = iRemIndex

    # if iThreadIndex_x >= dbg_trig:
    #     print('iSamplePerBlockIndex is:', iSamplePerBlockIndex)

    for col in range(2):    
        tmp = float32(0)
        for ant in range(ants):
            coeff = coeff_matrix[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][col][ant]
            data = data_matrix[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][ant]

            tmp += data * coeff

            # if iThreadIndex_x >= dbg_trig:
            #     print('iThreadIndex_x:', iThreadIndex_x, 'ant:', ant, 'tmp:', tmp)
            #     print('iThreadIndex_x:', iThreadIndex_x, 'ant:', ant, 'col:', col,'coeff:', coeff)
        
        # Copy computed weighted and summed ant samples to output
        if col == 0:
            # Computed sample is real component
            out[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][0] = tmp
        else:
            # Computed sample is imaginary component
             out[iBatchIndex][iPolIndex][iChanIndex][iBlockIndex][iSamplePerBlockIndex][1] = tmp    

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


class complex_mult_kernel:
    def complex_mult(self, data_matrix, coeff_matrix, out=None, stream=None):
        batches = data_matrix.shape[0]
        pols = data_matrix.shape[1]
        n_channel = data_matrix.shape[2]
        blocks = data_matrix.shape[3] 
        samples_per_block = data_matrix.shape[4]
        ants = data_matrix.shape[5]
        complexity = 2 # always

        # Reshape data to have ant real and imag data in one dimension
        data_matrix = data_matrix.reshape(batches, pols, n_channel, blocks, samples_per_block, (ants * complexity))

        # Create the data array - usually initialized some other way
        # data = np.ones(256)

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