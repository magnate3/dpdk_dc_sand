# https://python.hotexamples.com/site/file?hash=0x81b426ed3764536b43c0a4d0f978ba9722ea0d0cc098838d185452c4d7c597ab&fullName=scikits.cuda-master/test_cublas.py&project=alemagnani/scikits.cuda
# sgemm based on https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm  ->  2.7.1. cublas<t>gemm()

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from skcuda.cublas import *
from skcuda import cublas


def _bptrs(a):
    """
    Pointer array when input represents a batch of matrices.
    """
    return gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)

class cublas_SgemmBatched:
    def cublas_SgemmBatched(self, data_matrix, coeff_matrix, out=None, stream=None):
        # matrix -matrix multiplication: c=alpha*a*b+beta*c
        h = cublasCreate() # initialize cublas context 

        # l, m, n, k = 2, 6, 4, 5
        # l, m, n, k = 16, 1, 2, 64 * 2

        batches = data_matrix.shape[0]
        pols = data_matrix.shape[1]
        n_channel = data_matrix.shape[2]
        blocks = data_matrix.shape[3] 
        samples_per_block = data_matrix.shape[4]
        ants = data_matrix.shape[5]
        complexity = 2 # always

        # a = np.arange(1,(k*m*l+1),1,np.float32).reshape(l,k,m)
        # b = np.arange(1,(k*n*l+1),1,np.float32).reshape(l,n,k)
        # c = np.arange(1,(n*m*l+1),1,np.float32).reshape(l,n,m)
        # a_gpu = gpuarray.to_gpu(a.copy())
        coeff_matrix_gpu = gpuarray.to_gpu(coeff_matrix.copy())
        # c_gpu = gpuarray.to_gpu(c.copy())

        m, n, k = 1, 2, ants * 2
        l = batches * pols * n_channel * blocks * samples_per_block
        # l= 16
        lda = data_matrix.shape[5]
        ldb = coeff_matrix.shape[2]

        ldout = out.shape[5]
        c = np.zeros(l*2,np.float32).reshape(l,n,m)
        c_gpu = gpuarray.to_gpu(c.copy())

        alpha = np.float32(1.0)
        beta = np.float32(0.0)

        # data_matrix.dtype=np.float32
        a_arr = _bptrs(data_matrix.reshape(l,ants*2,1).astype(np.float32))
        b_arr = _bptrs(coeff_matrix_gpu)
        c_arr = _bptrs(c_gpu)
        # c_arr = _bptrs(out)
        
        # a = np.arange(1,(k*m*l+1),1,np.float32).reshape(l,k,m)
        # a_gpu = gpuarray.to_gpu(a.copy())
        # a_arr = _bptrs(a_gpu)

        cublas.cublasSgemmBatched(h, 'n','n', m, n, k, alpha, a_arr.gpudata, lda, b_arr.gpudata, ldb, beta, c_arr.gpudata, ldout, l)
        # print(c_gpu.get())
        # print(out.get())

        out = c_gpu.reshape(batches, pols, n_channel, blocks, samples_per_block, 2).astype(np.uint8)
        print(out.get())
        # TODO: Check data at this point

        # cublas.cublasSgemmBatched(h, 'n','n', m, n, k, alpha, a_arr.gpudata, lda, b_arr.gpudata, ldb, beta, c_arr.gpudata, ldout, l)
        
        # print(c_gpu.get())
