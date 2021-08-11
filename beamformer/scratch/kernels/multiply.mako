/*
 *  This kernel aims to carry out the multiplication.
 */
// Includes
#include <stdint.h>
#include <stdlib.h>
#include <sys/cdefs.h>

<%include file="/port.mako"/>

#define iThreadDBG 1

__global__
void multiply(uint16_t *pu16Array, uint16_t *pu16ArrayScaled, uint16_t scale)
{
    // 1. Declare indices used for reorder
    int iThreadIndex_x = blockIdx.x * blockDim.x + threadIdx.x;
    int yindex = blockIdx.y;

    // - Declaring in their order of dimensionality for the new matrix
    int iMatrixStride_y;
    
    // 2. Calculate indices for reorder
    iMatrixStride_y = yindex;

    // 3. Perform the multiplication
    uint16_t u16InputSample;
    uint16_t u16SampleScaled;

    // 3.1. Read out from the original arrayNo 
    u16InputSample = *(pu16Array + iThreadIndex_x + iMatrixStride_y);
        
    // 3.2. Store at its reordered index
    *(pu16ArrayScaled + iMatrixStride_y) = u16InputSample * scale;
    u16SampleScaled = *(pu16ArrayScaled + iMatrixStride_y);

    printf("Mult: iThreadIndex_x: %d blockIdx.x: %d blockDim.x: %d threadIdx.x: %d   InputSample:%d  Scale: %d  ScaledSample:%d\n", iThreadIndex_x, blockIdx.x, blockDim.x, threadIdx.x, u16InputSample, scale, u16SampleScaled);
    ## if((iThreadIndex_x == iThreadDBG)&(blockIdx.y == 0)){
    ##     printf("Mult: iThreadIndex_x: %d InputSample:%d ScaledSample:%d\n", iThreadIndex_x, u16InputSample, u16SampleScaled);         
    ## }
}
