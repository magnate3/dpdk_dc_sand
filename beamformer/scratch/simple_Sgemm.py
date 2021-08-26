# sgemm based on https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm  ->  2.7.1. cublas<t>gemm()

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.cublas as cublas

class simple_sgemm:
    def sgemm(a=None, b=None, out=None, stream=None):
        """Matrix multiplication using CUBLAS."""
        # note: we swap the order of a and b and swap the transposes because
        # cublas assumes column-major ordering
        transa = "n"
        transb = "n"
        alpha = 1.0
        beta = 0.0

        m=6;n=4;k=5
        a=np.arange(1,31,1,np.float32).reshape(k,m)
        b=np.arange(1,21,1,np.float32).reshape(n,k)
        c=np.arange(1,25,1,np.float32).reshape(n,m)
        a_gpu = gpuarray.to_gpu(a.copy())
        b_gpu = gpuarray.to_gpu(b.copy())
        c_gpu = gpuarray.to_gpu(c.copy())

        a0 = a.shape[0]
        a1 = a.shape[1]
        b1 = b.shape[1]
        lda = a1
        ldb = b1
        ldout = c.shape[1]

        handle = cublas.cublasCreate()

        gemm = cublas.cublasSgemm
    
        gemm(handle, transb, transa, m, n, k,
            alpha, a_gpu.gpudata, lda, b_gpu.gpudata, ldb, beta, c_gpu.gpudata,
            ldout)
    
        out = c_gpu.get()
        print(a)
        print(b)
        return out

C = simple_sgemm.sgemm()
print(C)