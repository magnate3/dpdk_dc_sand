import numpy as np
import pkg_resources
from skcuda.cublas import *
from katsdpsigproc import accel, cuda
import pycuda.gpuarray as gpuarray
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup
from beamforming.complex_mult_kernel import complex_mult_kernel
from beamforming.cublas_SgemmBatched import cublas_SgemmBatched

class MultiplyTemplate:
    def __init__(
        self, context: cuda.Context, n_ants: int, n_channels: int, n_samples_per_channel: int, n_batches: int
    ) -> None:
        self.context = context
        self.n_channels = n_channels
        self.n_samples_per_channel = n_samples_per_channel
        self.n_polarisations = 2  # Hardcoded to 2. No other values are supported
        self.n_batches = n_batches
        self.n_ants = n_ants
        self._sample_bitwidth = 8
        self.complexity = 2

        # This 128 is hardcoded in the original tensor core kernel. The reason it is set to this needs to be determined.
        self.n_samples_per_block = 128 // self._sample_bitwidth
        self.n_blocks = self.n_samples_per_channel // self.n_samples_per_block

        self.inputShape = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_polarisations, exact=True),
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_blocks, exact=True),
            accel.Dimension(self.n_samples_per_block, exact=True),
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(self.complexity, exact=True),
        )

        self.outputDataShape = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_polarisations, exact=True),
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_blocks, exact=True),
            accel.Dimension(self.n_samples_per_block, exact=True),
            accel.Dimension(self.complexity, exact=True),
        )

    def instantiate(self, command_queue: accel.AbstractCommandQueue, coeffs):
        return Multiply(self, command_queue, coeffs)

class Multiply(Operation):
    WGS = 32

    def __init__(self, template: MultiplyTemplate, command_queue: accel.AbstractCommandQueue, coeffs):
        super().__init__(command_queue)
        self.template = template
        self.coeffs = coeffs

        self.slots["inData"] = IOSlot(dimensions=self.template.inputShape, dtype=np.uint8)
        self.slots["outData"] = IOSlot(dimensions=self.template.outputDataShape, dtype=np.float32)

    def _run(self):
        with self.command_queue.context:
            # cublas_SgemmBatched.cublas_SgemmBatched(self, self.buffer("inData").buffer, self.coeffs, self.buffer("outData").buffer)
            complex_mult_kernel.complex_mult(self, self.buffer("inData").buffer, self.coeffs, self.buffer("outData").buffer)