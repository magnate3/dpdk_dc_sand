# https://python.hotexamples.com/site/file?hash=0x81b426ed3764536b43c0a4d0f978ba9722ea0d0cc098838d185452c4d7c597ab&fullName=scikits.cuda-master/test_cublas.py&project=alemagnani/scikits.cuda
# sgemm based on https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm  ->  2.7.1. cublas<t>gemm()

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from skcuda.cublas import *
from skcuda import cublas

def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.
    """
    
    return gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)


# matrix -matrix multiplication: c=alpha*a*b+beta*c
h = cublasCreate() # initialize cublas context 

l, m, n, k = 2, 6, 4, 5

a = np.arange(1,(k*m*l+1),1,np.float32).reshape(l,k,m)
b = np.arange(1,(k*n*l+1),1,np.float32).reshape(l,n,k)
c = np.arange(1,(n*m*l+1),1,np.float32).reshape(l,n,m)
a_gpu = gpuarray.to_gpu(a.copy())
b_gpu = gpuarray.to_gpu(b.copy())
c_gpu = gpuarray.to_gpu(c.copy())

a1 = a.shape[2]
b1 = b.shape[2]
lda = a1
ldb = b1
ldout = c.shape[2]

alpha = np.float32(1.0)
beta = np.float32(0.0)

a_arr = bptrs(a_gpu)
b_arr = bptrs(b_gpu)
c_arr = bptrs(c_gpu)

# TODO: fix for l > 1 
cublas.cublasSgemmBatched(h, 'n','n', m, n, k, alpha, a_arr.gpudata, lda, b_arr.gpudata, ldb, beta, c_arr.gpudata, ldout, l)

print(c_gpu.get())