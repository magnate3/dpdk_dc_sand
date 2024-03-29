"""
Module for beamformer multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
Provision for batched operations is included, i.e. reordering multiple sets of data (matrices) passed to the kernel
in a single array.
"""
import numpy as np
from beamforming.complex_mult_kernel import ComplexMultKernel
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
    [batch][polarizations][n_channels_per_stream][n_blocks][n_samples_per_channel][n_ants][complexity]

    The output beamforming buffer must have the shape:
    [batch][polarizations][n_channels_per_stream][n_blocks][n_samples_per_channel][complexity]

    The samples_per_channel index is split over two different indices. The outer index ranges from 0 to n_blocks and
    the inner index from 0 to n_samples_per_channel//n_blocks (i.e sample_per_block).

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
    n_channels_per_stream: int
        The number of frequency channels to be processed.
    n_samples_per_channel: int
        The number of samples per channel.
    n_beams: int
        Number of beams.
    n_batches: int
        The number of matrices to be reordered, a single data matrix = one batch.
    """

    def __init__(
        self,
        context: AbstractContext,
        n_ants: int,
        n_channels_per_stream: int,
        n_samples_per_channel: int,
        n_beams: int,
        n_batches: int,
    ) -> None:
        self.context = context
        self.n_ants = n_ants
        self.n_channels_per_stream = n_channels_per_stream
        self.n_samples_per_channel = n_samples_per_channel
        self.n_batches = n_batches
        self._sample_bitwidth = 8
        self.n_pols = 2  # Hardcoded to 2. No other values are supported
        self.complexity = 2
        self.beams = n_beams

        # This 128 is hardcoded in the original tensor core kernel. Likely to do with optimum thread-block size.
        # i.e. 4 warps totalling 128 threads per block.
        self.n_samples_per_block = 128 // self._sample_bitwidth
        self.n_blocks = self.n_samples_per_channel // self.n_samples_per_block
        self.length = (
            self.n_batches
            * self.n_pols
            * self.n_channels_per_stream
            * self.n_blocks
            * self.n_samples_per_block
        )

        self.input_data_dimensions = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_pols, exact=True),
            accel.Dimension(self.n_channels_per_stream, exact=True),
            accel.Dimension(self.n_blocks, exact=True),
            accel.Dimension(self.n_samples_per_block, exact=True),
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(self.complexity, exact=True),
        )

        self.output_data_dimensions = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_pols, exact=True),
            accel.Dimension(self.n_channels_per_stream, exact=True),
            accel.Dimension(self.n_blocks, exact=True),
            accel.Dimension(self.n_samples_per_block, exact=True),
            accel.Dimension(self.beams * self.complexity, exact=True),
        )

        self.coeff_data_dimensions = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_pols, exact=True),
            accel.Dimension(self.n_channels_per_stream, exact=True),
            accel.Dimension(self.n_ants * 2, exact=True),
            accel.Dimension(self.beams * 2, exact=True),
        )

    def instantiate(self, command_queue: accel.AbstractCommandQueue):
        """Initialise the complex multiplication class."""
        return MatrixMultiply(self, command_queue)


class MatrixMultiply(Operation):
    """Class for beamform complex multiplication.

    .. rubric:: Slots
    inData: (batches, n_pols, n_channels_per_stream, n_blocks, n_samples_per_block, n_ants, complexity), uint8
        Input reordered channelised data.
    outData: (batches, n_pols, n_channels_per_stream, n_blocks, n_samples_per_block, complexity), float32
        Beamformed data.
    inCoeffs: (batches, n_pols, n_channels_per_stream, n_blocks, n_samples_per_block, complexity, n_ants, 2), float32
        Beamforming coefficients.

    Parameters
    ----------
    template: MatrixMultiplyTemplate
        Template for multiplication class
    command_queue: accel.AbstractCommandQueue
        CUDA command queue
    """

    def __init__(
        self,
        template: MatrixMultiplyTemplate,
        command_queue: accel.AbstractCommandQueue,
    ):
        super().__init__(command_queue)
        self.template = template

        self.slots["inData"] = IOSlot(
            dimensions=self.template.input_data_dimensions, dtype=np.uint8
        )
        self.slots["outData"] = IOSlot(
            dimensions=self.template.output_data_dimensions, dtype=np.float32
        )
        self.slots["inCoeffs"] = IOSlot(
            dimensions=self.template.coeff_data_dimensions, dtype=np.float32
        )

    def _run(self):
        """Run the beamform computation."""
        with self.command_queue.context:
            ComplexMultKernel.complex_mult(
                self,
                self.buffer("inData").buffer,
                self.buffer("inCoeffs").buffer,
                self.buffer("outData").buffer,
            )
