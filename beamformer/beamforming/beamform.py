"""
Module for beamformer multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
Provision for batched operations is included, i.e. reordering multiple sets of data (matrices) passed to the kernel
in a single array.
"""
import numpy as np
from beamforming.complex_mult_kernel import complex_mult_kernel
from beamforming.cublas_SgemmBatched import cublas_SgemmBatched
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext
from katsdpsigproc.accel import IOSlot, Operation


class MultiplyTemplate:
    """
    Template class for beamform multiplication.

    This class specifies the shape of the input sample and output beamformed data.
    The parameters specified are used to determine the shape of the buffers.

    It is worth noting these matrices follow the C convention, with the fastest-changing dimension being
    the last on the list.
    The input sample buffer must have the shape:
    [batch][polarizations][n_channels][n_blocks][samples_per_block][n_ants][complexity]

    The output beamforming buffer must have the shape:
    [batch][polarizations][n_channels][n_blocks][samples_per_block][complexity]

    The samples_per_channel index is split over two different indices. The outer index ranges from 0 to n_blocks and
    the inner index from 0 to samples_per_channel//n_blocks (i.e sample_per_block).

    Each input element is a complex 8-bit integer sample.
    Each output sample is a complex 32b float.

    Parameters
    ----------
    context: AbstractContext
        The GPU device's context provided by katsdpsigproc's abstraction of PyCUDA.
        A context is associated with a single device and 'owns' all memory allocations.
        For the purposes of this python module the CUDA context is required.
    n_batches: int
        The number of matrices to be reordered, a single data matrix = one batch.
    pols: int
        Number of polarisations. Always 2.
    n_channels: int
        The number of frequency channels to be processed.
    n_blocks: int
        The number of blocks that each channels set of samples are divided into.
    samples_per_block: int
        The number of time samples to be processed per block.
    n_ants: int
        The number of antennas that will be used in beamforming. Each antennas is expected to produce two polarisations.
    """

    def __init__(
        self, context: AbstractContext, n_ants: int, n_channels: int, n_samples_per_channel: int, n_batches: int
    ) -> None:
        """Initialise the MultiplyTemplate class."""
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

    def instantiate(self, command_queue: accel.AbstractCommandQueue, coeffs, test_id):
        """Initialise the complex multiplication class."""
        return Multiply(self, command_queue, coeffs, test_id)


class Multiply(Operation):
    """Class for beamform complex multiplication.

    Parameters
    ----------
    template: MultiplyTemplate
        Template for multiplication class
    coeffs: nd.array of type float
        Coefficianets for beamforming computation.
    test_id: string
        ID of the computation to run. This will be removed and is only for testing.
    """

    def __init__(self, template: MultiplyTemplate, command_queue: accel.AbstractCommandQueue, coeffs, test_id):
        """Initialise the Multiply class."""
        super().__init__(command_queue)
        self.template = template
        self.coeffs = coeffs
        self.test_id = test_id

        self.slots["inData"] = IOSlot(dimensions=self.template.inputShape, dtype=np.uint8)
        self.slots["outData"] = IOSlot(dimensions=self.template.outputDataShape, dtype=np.float32)

    def _run(self):
        """Run the beamform computation."""
        with self.command_queue.context:
            if self.test_id == "sgemm":
                cublas_SgemmBatched.cublas_SgemmBatched(
                    self, self.buffer("inData").buffer, self.coeffs, self.buffer("outData").buffer
                )
            if self.test_id == "kernel":
                complex_mult_kernel.complex_mult(
                    self, self.buffer("inData").buffer, self.coeffs, self.buffer("outData").buffer
                )
