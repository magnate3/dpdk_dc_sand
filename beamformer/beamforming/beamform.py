import numpy as np
import pkg_resources
from skcuda.cublas import *
from katsdpsigproc import accel, cuda
import pycuda.gpuarray as gpuarray
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup
import cublas_mult


def cublas_dot(self, a, b, out=None, transpose_a=False, transpose_b=False,
                stream=None):
    """Matrix multiplication using CUBLAS."""
    aa = a.astype(np.float32)
    bb = b.astype(np.float32)
    dtype = a.dtype
    
    # if transpose_a:
    #     a1, a0 = a.shape
    # else:
    #     a0, a1 = a.shape
    
    # if transpose_b:
    #     b1, b0 = b.shape
    # else:
    #     b0, b1 = b.shape

    if transpose_a:
        a1, a0 = a.shape
    else:
        a0 = self.template.m
        a1 = self.template.n
    
    if transpose_b:
        b1, b0 = b.shape
    else:
        b0 = self.template.n
        b1 = self.template.k
    
    assert a1 == b0
    
    if out is None:
        out = gpuarray.zeros((a0, b1), dtype=np.float32)
    
    a_gpu = gpuarray.to_gpu(self.template.a.T.copy()) # mxk matr.on the device 
    b_gpu = gpuarray.to_gpu(self.template.b.T.copy()) # kxn matr.on the device 
    
    #assert a.dtype == b.dtype == out.dtype
    
    # note: we swap the order of a and b and swap the transposes because
    # cublas assumes column-major ordering
    transa = "t" if transpose_a else "n"
    transb = "t" if transpose_b else "n"
    alpha = dtype.type(1.0)
    beta = dtype.type(0.0)
    lda = a0 if transpose_a else a1
    ldb = b0 if transpose_b else b1
    ldout = b1
    
    handle = cublasCreate()
    if stream is not None:
        # note: this assumes that the stream is set to the default stream to
        # start
        cublas.cublasSetStream(handle, stream.handle)

    # if dtype == np.float32:
    gemm = cublasSgemm
    # else:
    #     gemm = cublasDgemm
    
    # gemm(handle, transb, transa, b1, a0, a1,
    #     alpha, b.gpudata, ldb, a.gpudata, lda, beta, out.gpudata,
    #     ldout)

    gemm(handle, transb, transa, b1, a0, a1,
        alpha, b_gpu.gpudata, ldb, a_gpu.gpudata, lda, beta, out.gpudata,
        ldout)

    print(out)
    return out

class MultiplyTemplate:
    def __init__(self, context: cuda.Context) -> None:
        self.context = context

        # self.h = cublasCreate() # initialize cublas context
        self.alpha = np. float32 (1.0) # scalar alpha
        self.beta = np. float32 (1.0) # scalar beta
        #self.cublas_mult = cublas_mult.cublas_mult(queue)

        self.m = 6
        self.n = 4
        self.k = 5 # a - mxk matrix , b - kxn matrix , c - mxn matrix
        
        self.a = np.arange(11 ,41 ,1 , np.float32).reshape(self.k, self.m).T # mxk matrix
        self.b = np.arange(11 ,31 ,1 , np.float32).reshape(self.n, self.k).T # kxn matrix
        self.c = np.arange(11 ,35 ,1 , np.float32).reshape(self.n, self.m).T # mxn matrix

    def instantiate(self, command_queue: accel.AbstractCommandQueue, inshape, outshape):
        return Multiply(self, command_queue, inshape, outshape)

class Multiply(Operation):
    WGS = 32

    def __init__(self, template: MultiplyTemplate, command_queue: accel.AbstractCommandQueue, inshape, outshape):
        super().__init__(command_queue)
        self.template = template

        # if stream is not None:
        #     # note: this assumes that the stream is set to the default stream to
        #     # start
        #     cublas.cublasSetStream(misc._global_cublas_handle, stream.handle)

        # self.scale = np.uint16(scale)

        self.slots["inData"] = IOSlot(dimensions=inshape, dtype=np.uint8)
        self.slots["outData"] = IOSlot(dimensions=outshape, dtype=np.uint8)

        # self.scale = np.uint16(scale)
        # self.slots["inData"] = IOSlot(dimensions=shape, dtype=np.uint8)
        # self.slots["outData"] = IOSlot(dimensions=shape, dtype=np.uint8)


        # program = build(
        #     queue.context,
        #     "kernels/beamform_kernel.mako",
        #     {
        #         "scale": scale,
        #     },
        #     extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        # )

    def _run(self):

        with self.command_queue.context:
            cublas_dot(self, self.buffer("inData").buffer, self.buffer("outData").buffer)