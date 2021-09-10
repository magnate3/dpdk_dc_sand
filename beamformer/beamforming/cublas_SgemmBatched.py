"""
Module performing beamforming multiplication using cublas_SgemmBached.

This implementation performs the complex multiplication in batches.

NOTE: This implemetation has a memory leak and causes the GPU to run out of memory.

TODO: DEBUG!!

Some Sources:
https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/
https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
https://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.cublas.cublasSgemmBatched.html#skcuda.cublas.cublasSgemmBatched
https://python.hotexamples.com/site/file?hash=0x81b426ed3764536b43c0a4d0f978ba9722ea0d0cc098838d185452c4d7c597ab&fullName=scikits.cuda-master/test_cublas.py&project=alemagnani/scikits.cuda
sgemm based on https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm  ->  2.7.1. cublas<t>gemm()
https://stackoverflow.com/questions/16376804/clarification-of-the-leading-dimension-in-cublas-when-transposing
"""

import numpy as np
import pycuda.gpuarray as gpuarray
from skcuda import cublas
from skcuda.cublas import cublasCreate


def _bptrs(a):
    """Pointer array when input represents a batch of matrices."""
    return gpuarray.arange(a.ptr, a.ptr + a.shape[0] * a.strides[0], a.strides[0], dtype=cublas.ctypes.c_void_p)


class cublas_SgemmBatched:
    """Class for cublas_SgemmBatched.

    Parameters
    ----------
    data_matrix: nd.array[np.uint8]
        Data matrix on reordered data
    coeff_matrix: nd.array[np.float32]
        Coefficients for beamforming computation.
    out: nd.array[np.float32]
        Complex multipication product for beamforming computation.
    """

    def cublas_SgemmBatched(self, data_matrix, coeff_matrix, out=None, stream=None):
        """Initialise the coefficient generation class.

        Computation to be performed: matrix - matrix multiplication:
        c=alpha*a*b+beta*c
        """
        h = cublasCreate()  # initialize cublas context and get a handle

        batches = data_matrix.shape[0]
        pols = data_matrix.shape[1]
        n_channel = data_matrix.shape[2]
        blocks = data_matrix.shape[3]
        samples_per_block = data_matrix.shape[4]
        ants = data_matrix.shape[5]

        alpha = np.float32(1.0)
        beta = np.float32(0.0)

        m, n, k = 1, 2, ants * 2
        total_length = batches * pols * n_channel * blocks * samples_per_block

        coeff_matrix_gpu = gpuarray.to_gpu(coeff_matrix.copy())

        # Setup leading dimensions.
        # Note: The leading dimension refers to the length of the first dimension in the array.
        # For this, the first dimension is rows as it is stored in row-major order.
        lda = data_matrix.reshape(total_length, ants * 2, 1).astype(np.float32).shape[2]
        ldb = coeff_matrix.shape[2]
        ldout = out.reshape(total_length, n, m).shape[2]

        data_reshape = data_matrix.reshape(total_length, ants * 2, 1).astype(np.float32)
        a_arr = _bptrs(data_reshape)
        b_arr = _bptrs(coeff_matrix_gpu)
        c_arr = _bptrs(out.reshape(total_length, n, m))

        cublas.cublasSgemmBatched(
            h,
            "n",
            "n",
            m,
            n,
            k,
            alpha,
            a_arr.gpudata,
            lda,
            b_arr.gpudata,
            ldb,
            beta,
            c_arr.gpudata,
            ldout,
            total_length,
        )
