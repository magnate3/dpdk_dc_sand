import pycuda.autoinit
import pycuda.gpuarray as gpuarray
# import pycuda.driver as drv
import numpy as np
import skcuda.cublas as cublas
from numba import cuda

class cublas_gemm:
    def cublas_gemm(self, a, b, out=None, stream=None):
        """Matrix multiplication using CUBLAS."""
        dtype = a.dtype

        # batches, pols, n_channels, n_blocks, n_samples_per_block, n_ants, complexity = a.shape
        # coeffs_shape = (np.shape(b))

        # note: we swap the order of a and b and swap the transposes because
        # cublas assumes column-major ordering
        transa = "n"
        transb = "n"
        alpha = dtype.type(1.0)
        beta = dtype.type(0.0)
        # lda = n_ants
        # ldb = n_ants * complexity

        m=6;n=4;k=5
        a=np.arange(11,41,1,np.float32).reshape(k,m).T
        b=np.arange(11,31,1,np.float32).reshape(n,k).T
        c=np.arange(11,35,1,np.float32).reshape(n,m).T
        a_gpu = gpuarray.to_gpu(a.T.copy())
        b_gpu = gpuarray.to_gpu(b.T.copy())
        c_gpu = gpuarray.to_gpu(c.T.copy())

        a0 = a.shape[0]
        a1 = a.shape[1]
        b1 = b.shape[1]
        lda = a1
        ldb = b1
        ldout = c.shape[0]

        handle = cublas.cublasCreate()
        if stream is not None:
            # note: this assumes that the stream is set to the default stream to
            # start
            cublas.cublasSetStream(handle, stream.handle)

        if dtype == np.float32:
            gemm = cublas.cublasSgemm
        else:
            gemm = cublas.cublasDgemm
    
        gemm(handle, transb, transa, b1, a0, a1,
            alpha, b_gpu.gpudata, ldb, a_gpu.gpudata, lda, beta, c_gpu.gpudata,
            ldout)
    
        return out

# A_col = 10000
# A_Row = 1000
# A = np.random.randint(10, size=(A_Row,A_col)).astype(np.float32)
# B = np.random.randint(10, size=(A_col,A_Row)).astype(np.float32)

# m, k = A.shape
# k, n = B.shape

# # Copy to GPU memory
# A_gpu = gpuarray.to_gpu(A)
# B_gpu = gpuarray.to_gpu(B)
# C_gpu = gpuarray.empty((m, n), np.float32)

# C = cublas_gemm.cublas_gemm(A_gpu,B_gpu,C_gpu)
# print(C)