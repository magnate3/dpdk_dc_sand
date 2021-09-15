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


class MatrixMultiplyTemplate:
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
    n_ants: int
        The number of antennas that will be used in beamforming. Each antennas is expected to produce two polarisations.
    n_channels: int
        The number of frequency channels to be processed.
    n_samples_per_channel: int
        The number of samples per channel.
    batches: int
        The number of matrices to be reordered, a single data matrix = one batch.
    """

    def __init__(
        self,
        context: AbstractContext,
        n_ants: int,
        n_channels: int,
        n_samples_per_channel: int,
        batches: int,
        test_id,
    ) -> None:
        self.context = context
        self.n_ants = n_ants
        self.n_channels = n_channels
        self.n_samples_per_channel = n_samples_per_channel
        self.batches = batches
        self._sample_bitwidth = 8
        self.n_pols = 2  # Hardcoded to 2. No other values are supported
        self.complexity = 2
        self.test_id = test_id

        # This 128 is hardcoded in the original tensor core kernel. Likely to do with optimum thread-block size.
        # i.e. 4 warps totalling 128 threads per block.
        self.n_samples_per_block = 128 // self._sample_bitwidth
        self.n_blocks = self.n_samples_per_channel // self.n_samples_per_block
        self.length = self.batches * self.n_pols * self.n_channels * self.n_blocks * self.n_samples_per_block

        self.input_data_dimensions = (
            accel.Dimension(self.batches, exact=True),
            accel.Dimension(self.n_pols, exact=True),
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_blocks, exact=True),
            accel.Dimension(self.n_samples_per_block, exact=True),
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(self.complexity, exact=True),
        )

        self.output_data_dimensions = (
            accel.Dimension(self.batches, exact=True),
            accel.Dimension(self.n_pols, exact=True),
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_blocks, exact=True),
            accel.Dimension(self.n_samples_per_block, exact=True),
            accel.Dimension(self.complexity, exact=True),
        )

        if test_id == "kernel":
            self.coeff_data_dimensions = (
                accel.Dimension(self.batches, exact=True),
                accel.Dimension(self.n_pols, exact=True),
                accel.Dimension(self.n_channels, exact=True),
                accel.Dimension(self.n_blocks, exact=True),
                accel.Dimension(self.n_samples_per_block, exact=True),
                accel.Dimension(self.complexity, exact=True),
                accel.Dimension(self.n_ants * self.complexity, exact=True),
            )
        elif test_id == "sgemm":
            self.coeff_data_dimensions = (
                accel.Dimension(self.length, exact=True),
                accel.Dimension(self.complexity, exact=True),
                accel.Dimension(self.n_ants * self.complexity, exact=True),
            )

    def instantiate(self, command_queue: accel.AbstractCommandQueue, test_id):
        """Initialise the complex multiplication class."""
        return MatrixMultiply(self, command_queue, test_id)


class MatrixMultiply(Operation):
    """Class for beamform complex multiplication.

    Parameters
    ----------
    template: MultiplyTemplate
        Template for multiplication class
    command_queue: accel.AbstractCommandQueue
        CUDA command queue
    coeffs: nd.array[np.float32]
        Coefficianets for beamforming computation.
    test_id: string
        ID of the computation to run. This will be removed and is only for testing.
    """

    def __init__(self, template: MatrixMultiplyTemplate, command_queue: accel.AbstractCommandQueue, test_id):
        super().__init__(command_queue)
        self.template = template
        self.test_id = test_id

        self.slots["inData"] = IOSlot(dimensions=self.template.input_data_dimensions, dtype=np.uint8)
        self.slots["outData"] = IOSlot(dimensions=self.template.output_data_dimensions, dtype=np.float32)
        self.slots["inCoeffs"] = IOSlot(dimensions=self.template.coeff_data_dimensions, dtype=np.float32)

    def _run(self):
        """Run the beamform computation."""
        with self.command_queue.context:
            if self.test_id == "sgemm":
                cublas_SgemmBatched.cublas_SgemmBatched(
                    self, self.buffer("inData").buffer, self.buffer("inCoeffs").buffer, self.buffer("outData").buffer
                )
            if self.test_id == "kernel":
                complex_mult_kernel.complex_mult(
                    self, self.buffer("inData").buffer, self.buffer("inCoeffs").buffer, self.buffer("outData").buffer
                )
