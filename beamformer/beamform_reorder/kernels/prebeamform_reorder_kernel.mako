/*
 *  This kernel aims to carry out the reorder functionality required by katxgpu beamformer.
 *  This GPU-side reorder makes provision for batched operations (i.e. reordering batches of matrices),
 *  and transforms a 1D block of data in the following matrix format:
 *  - uint16_t [n_batches][n_antennas][n_channels][n_samples_per_channel][polarizations]
 *    transposed to
 *    uint16_t [n_batches][polarizations][n_channels][n_blocks][samples_per_channel//n_blocks][n_ants]
 *  - Typical values for the dimensions
 *      - n_antennas (a) = 64
 *      - n_channels (c) = 128
 *      - n_samples_per_channel (t) = 256
 *      - polarisations (p) = 2, always
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
#define NR_CHANNELS ${n_channels}
#define NR_SAMPLES_PER_CHANNEL ${n_samples_per_channel}
#define NR_POLARISATIONS ${n_polarisations}
#define NR_SAMPLES_PER_BLOCK ${n_samples_per_block}

// *** DEBUG ***
#define iThreadDBG 2 //129023 //130048 //104448
#define iprintDBG 130550 //129023 //130048 //104448
#define bfNewIdxDBG 23352

__global__
void prebeamform_reorder(uint16_t *pu16Array, uint16_t *pu16ArrayReordered)
{
    // 1. Declare indices used for reorder
    int iThreadIndex_x = blockIdx.x * blockDim.x + threadIdx.x;
    int yindex = blockIdx.y;

    int iChanIndex, iTimeIndex, iPolIndex;

    // *** DEBUG ***
    if((iThreadIndex_x == iThreadDBG)&(blockIdx.y == 0)){
        printf("Debug 1\n");
        printf("iThreadIndex_x: :%d\n", iThreadIndex_x);
        printf("blockDim.x: %d\n", blockDim.x);
        printf("blockIdx.x: %d\n", blockIdx.x);
        printf("blockIdx.z: %d\n", blockIdx.z);        
        printf("Thread: %d\n", threadIdx.x);
        printf("\n");
    }

    // - Declaring in their order of dimensionality for the new matrix
    int iNewIndex, iNewChanOffset, iTimeOuterOffset, iNewAntOffset, iNewPolOffset;
    int iTimeOuterIndex, iTimeInnerIndex, iMatrixStride_y;
    
    // 2. Calculate indices for reorder
    // 2.1. Calculate 'current'/original indices for each dimension
    //      - Matrix Stride should be the same value for Original and Reordered matrices
    iMatrixStride_y = yindex * NR_ANTENNAS * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS;
    int iAntIndex = iThreadIndex_x / (NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);
    int iRemIndex = iThreadIndex_x % (NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);

    // *** DEBUG ***
    if((iThreadIndex_x == iThreadDBG)&(blockIdx.y == 0)){
        printf("Debug iAnt\n");
        printf("iAntIndex: %d\n", iAntIndex);
        printf("iRemIndex: %d\n", iRemIndex);
        printf("\n");
    }

    // *** DEBUG ***
    if((iThreadIndex_x == iThreadDBG)&(blockIdx.y == 0)){
        printf("Debug 2\n");
        printf("iMatrixStride_y: %d\n", iMatrixStride_y);
        printf("NR_ANTENNAS: %d\n", NR_ANTENNAS);
        printf("NR_CHANNELS: %d\n", NR_CHANNELS);
        printf("NR_SAMPLES_PER_CHANNEL: %d\n", NR_SAMPLES_PER_CHANNEL);      
        printf("NR_POLARISATIONS: %d\n", NR_POLARISATIONS);   
        printf("iAntIndex: %d\n", iAntIndex); 
        printf("iRemIndex: %d\n", iRemIndex);
        printf("\n");
    }
    
    iChanIndex = iRemIndex / (NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);
    iRemIndex = iRemIndex % (NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);

    // *** DEBUG ***
    if((iThreadIndex_x == iThreadDBG)&(blockIdx.y == 0)){
        printf("Debug 3\n"); 
        printf("iChanIndex: %d\n", iChanIndex); 
        printf("iRemIndex: %d\n", iRemIndex);
        printf("\n");
    }

    iTimeIndex = iRemIndex / NR_POLARISATIONS;
    iRemIndex = iRemIndex % NR_POLARISATIONS;
    // 0 = Even = Pol-0, 1 = Odd = Pol-1
    iPolIndex = iRemIndex;

    // *** DEBUG ***
    if((iThreadIndex_x == iThreadDBG)&(blockIdx.y == 0)){
        printf("Debug 4\n"); 
        printf("iTimeIndex: %d\n", iTimeIndex); 
        printf("iPolIndex: %d\n", iRemIndex);
        printf("\n"); 
    }

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

    // *** DEBUG ***
    if((iThreadIndex_x == iThreadDBG)&(blockIdx.y == 0)){
        printf("Debug 6\n"); 
        printf("iTimeIndex: %d\n", iTimeIndex);         
        printf("bfNewAntIdx: %d\n", bfNewAntIdx); 
        printf("bfNewChanIdx: %d\n", bfNewChanIdx);
        printf("bfNewPolIdx: %d\n", bfNewPolIdx);
        printf("bfNewTimeOuter: %d\n", bfNewTimeOuter);  
        printf("bfNewTimeInner: %d\n", bfNewTimeInner);
        printf("bfNewIdx: %d\n", bfNewIdx);
        printf("\n"); 
    }

    // 3. Perform the reorder (where necessary)
    uint16_t u16InputSample;

    // *** DEBUG ***
    uint16_t u16TempSample;

    if (iThreadIndex_x < (NR_ANTENNAS * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS))
    {
        // 3.1. Read out from the original arrayNo 
        u16InputSample = *(pu16Array + iThreadIndex_x + iMatrixStride_y);

        // 3.2. Store at its reordered index
        *(pu16ArrayReordered + bfNewIdx + iMatrixStride_y) = u16InputSample;

        // *** DEBUG ***
        u16TempSample = *(pu16ArrayReordered + bfNewIdx + iMatrixStride_y);

        // *** DEBUG ***
        if((iThreadIndex_x == iThreadDBG)&(blockIdx.y == 0)){
            printf("Debug 7\n"); 
            printf("Input Idx %d\n", iThreadIndex_x + iMatrixStride_y);
            printf("Reorder Idx %d\n", bfNewIdx + iMatrixStride_y);
            printf("iMatrixStride_y %d\n", iMatrixStride_y);
            printf("u16InputSample: %d\n", u16InputSample); 
            printf("\n"); 
        }

        // *** DEBUG ***
        if((u16InputSample == 8525)&(blockIdx.y == 0)){
            printf("u16InputSample Debug\n"); 
            printf("bfNewIdx: %d\n", bfNewIdx);
            printf("pu16ArrayReordered: %d\n", pu16ArrayReordered);
            printf("Input Idx: %d\n", iThreadIndex_x + iMatrixStride_y);
            printf("Reorder Idx: %d\n", bfNewIdx + iMatrixStride_y);
            printf("iMatrixStride_y: %d\n", iMatrixStride_y);
            printf("u16InputSample: %d\n", u16InputSample); 
            printf("u16TempSample: %d\n", u16TempSample); 
            printf("\n"); 
            printf("bfNewAntIdx: %d\n", bfNewAntIdx); 
            printf("bfNewChanIdx: %d\n", bfNewChanIdx);             
            printf("bfNewPolIdx: %d\n", bfNewPolIdx);            
            printf("bfNewTimeOuter: %d\n", bfNewTimeOuter);
            printf("bfNewTimeInner: %d\n", bfNewTimeInner);            
            printf("\n"); 
        }
    }
}
