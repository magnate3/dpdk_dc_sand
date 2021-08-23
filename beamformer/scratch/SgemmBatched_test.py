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

# m=6;n=4;k=5 # a - mxk matrix, b - kxn matrix, 

# a=np.arange(1,31,1,np.float32).reshape(k,m).T 
# b=np.arange(1,21,1,np.float32).reshape(n,k).T 
#c=np.arange(11,35,1,np.float64).reshape(n,m).T
# c=np.ones([m,n], dtype=np.float32)

# alpha = np.float32(1.0)
# beta = np.float32(1.0)

# a_gpu = gpuarray.to_gpu(a.T.copy()) # mxk matr.on the device 
# b_gpu = gpuarray.to_gpu(b.T.copy()) # kxn matr.on the device 
# c_gpu = gpuarray.to_gpu(c.T.copy()) # mxn matr.on the device
# cb_gpu = gpuarray.to_gpu(c.T.copy()) # mxn matr.on the device 
#h1 = cublasCreate() # initialize cublas context 

# matrix -matrix multiplication: c=alpha*a*b+beta*c
# syntax:
# cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, # B, ldb, beta, C, ldc)
# cublas.cublasSgemm(h1, 'n', 'n', a.shape[0], b.shape[1], a.shape[1], alpha, a_gpu.gpudata, a.shape[0], b_gpu.gpudata, b.shape[0], beta, c_gpu.gpudata, c.shape[0])
# initialize pycuda # import gpuarray # import numpy # import scikit-cuda cublas c - mxn matrix # mxk matrix # kxn matrix # mxn matrix # scalar alpha # scalar beta

h2 = cublasCreate() # initialize cublas context 
# batchCount = 0
# l, m, k, n = 11, 7, 5, 3
# A = np.random.rand(l, m, k).astype(np.float32)
# B = np.random.rand(l, k, n).astype(np.float32)
# # A = np.arange(1,(k*m*l+1),1,np.float32).reshape(k,m,l).T 
# # B = np.arange(1,(k*n*l+1),1,np.float32).reshape(n,k,l).T 

# a_gpu = gpuarray.to_gpu(A)
# b_gpu = gpuarray.to_gpu(B)
# c_gpu = gpuarray.empty((l, m, n), np.float32)
# a_arr = bptrs(a_gpu)
# b_arr = bptrs(b_gpu)
# c_arr = bptrs(c_gpu)
# cublas.cublasSgemmBatched(h2, 'n', 'n', n, m, k, alpha, b_arr.gpudata, n, a_arr.gpudata, k, beta, c_arr.gpudata, , l)

# print("a: \n", A[0])
# print("b: \n", B[0])

# #c1=c_gpu.get().T 
# #print("a*b+c: \n", c1)

# cb=c_gpu.get().T 
# print("a*b+c: \n", cb)
# cublasDestroy(h2)


l, m, n, k = 2, 6, 4, 5

# A = np.random.rand(l, m, k).astype(np.float32)
# B = np.random.rand(l, k, n).astype(np.float32)
        
# C_res = np.einsum('nij,njk->nik', A, B)
        
# a_gpu = gpuarray.to_gpu(A)
# b_gpu = gpuarray.to_gpu(B)
# c_gpu = gpuarray.empty((l, m, n), np.float32)
        

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
cublas.cublasSgemmBatched(h2, 'n','n', m, n, k, alpha, a_arr.gpudata, lda, b_arr.gpudata, ldb, beta, c_arr.gpudata, ldout, l)

print(c_gpu.get())

a = 1
# assert np.allclose(C_res, c_gpu.get())
# a = 1