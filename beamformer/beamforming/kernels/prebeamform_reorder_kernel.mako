/*
 *  This kernel aims to carry out the reorder functionality required by katxgpu beamformer.
 *  This GPU-side reorder makes provision for batched operations (i.e. reordering batches of matrices),
 *  and transforms a 1D block of data in the following matrix format:
 *  - uint16_t [n_batches][n_ants][n_channels_per_stream][n_samples_per_channel][polarizations]
 *    transposed to
 *    uint16_t [n_batches][polarizations][n_channels_per_stream][n_blocks][samples_per_channel//n_blocks][n_ants]
 *  - Typical values for the dimensions
 *      - n_ants (a) = 64
 *      - n_channels_per_stream (c) = 128
 *      - n_samples_per_channel (t) = 256
 *      - n_pols (p) = 2, always
 *      - times_per_block = 16
 *
 *   Currently, all dimension-strides are calculated within the kernel itself.
 *   - Granted, there are some redudancies/inefficiences in variable usage; however,
 *   - The kernel itself is operating as required.
 *
 *   \param[in]  pu16Array           Pointer to a pre-populated input data array. The input array is one-dimensional but stores
 *                                   multidimensional data according to the format described above.
 *   \param[out] pu16ArrayReordered  Pointer to the memory allocated for the reordered output data. Once more, this 1D output array
 *                                   represents multidimensional data in the format described above.
 */
// Includes
#include <stdint.h>
#include <stdlib.h>
#include <sys/cdefs.h>

<%include file="/port.mako"/>
// Defines, now using mako parametrisation
#define NR_ANTENNAS ${n_ants}
#define NR_CHANNELS ${n_channels_per_stream}
#define NR_SAMPLES_PER_CHANNEL ${n_samples_per_channel}
#define NR_POLARISATIONS ${n_pols}
#define NR_SAMPLES_PER_BLOCK ${n_samples_per_block}

__global__
void prebeamform_reorder(uint16_t *pu16Array, uint16_t *pu16ArrayReordered)
{
    // 1. Declare indices used for reorder
    int iThreadIndex_x = blockIdx.x * blockDim.x + threadIdx.x;
    int yindex = blockIdx.y;

    int iChanIndex, iTimeIndex, iPolIndex;

    // - Declaring in their order of dimensionality for the new matrix
    int iNewIndex, iNewChanOffset, iTimeOuterOffset, iNewAntOffset, iNewPolOffset;
    int iTimeOuterIndex, iTimeInnerIndex, iMatrixStride_y;

    // 2. Calculate indices for reorder
    // 2.1. Calculate 'current'/original indices for each dimension
    //      - Matrix Stride should be the same value for Original and Reordered matrices
    iMatrixStride_y = yindex * NR_ANTENNAS * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS;
    int iAntIndex = iThreadIndex_x / (NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);
    int iRemIndex = iThreadIndex_x % (NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);

    iChanIndex = iRemIndex / (NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);
    iRemIndex = iRemIndex % (NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);

    iTimeIndex = iRemIndex / NR_POLARISATIONS;
    iRemIndex = iRemIndex % NR_POLARISATIONS;
    // 0 = Even = Pol-0, 1 = Odd = Pol-1
    iPolIndex = iRemIndex;

    // 2.2. Calculate reordered matrix's indices and stride accordingly
    // New: prebeamform_reorder
    int bfNewAntIdx, bfNewChanIdx, bfNewPolIdx, bfNewIdx = 0;
    int bfNewTimeInner, bfNewTimeOuter = 0;

    bfNewAntIdx =  iAntIndex;

    bfNewChanIdx = iChanIndex * (NR_ANTENNAS * NR_SAMPLES_PER_CHANNEL);

    bfNewPolIdx = iPolIndex * NR_SAMPLES_PER_CHANNEL * NR_ANTENNAS * NR_CHANNELS;

    bfNewTimeOuter = (iTimeIndex/NR_SAMPLES_PER_BLOCK) * (NR_ANTENNAS * NR_SAMPLES_PER_BLOCK);

    bfNewTimeInner = (iTimeIndex % NR_SAMPLES_PER_BLOCK) * NR_ANTENNAS;

    bfNewIdx = bfNewPolIdx + bfNewChanIdx + bfNewTimeOuter + bfNewTimeInner + bfNewAntIdx;

    // 3. Perform the reorder (where necessary)
    uint16_t u16InputSample;

    if (iThreadIndex_x < (NR_ANTENNAS * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS))
    {
        // 3.1. Read out from the original arrayNo
        u16InputSample = *(pu16Array + iThreadIndex_x + iMatrixStride_y);

        // 3.2. Store at its reordered index
        *(pu16ArrayReordered + bfNewIdx + iMatrixStride_y) = u16InputSample;
    }
}
