#ifndef __BERAMFORMER_KERNELS_CUH__
#define __BERAMFORMER_KERNELS_CUH__

#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void cuda_hello();

__global__ void calculate_beamweights_naive(
                                struct timespec current_time, 
                                struct delay_vals_extended *dv, 
                                float* cplx_beamweights);

#endif