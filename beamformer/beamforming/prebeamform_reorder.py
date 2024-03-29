"""
Module wrapping the pre-correlation reorder Kernel.

The pre-correlation reorder kernel operates on a set of data with dimensions explained (and in its _kernel.mako file).
It makes provision for batched operations, i.e. reordering multiple sets of data (matrices) passed to the kernel
in a single array.
"""
import numpy as np
import pkg_resources
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext
from typing_extensions import Final


class PreBeamformReorderTemplate:
    """
    Template class for compiling different variations of the pre-beamform reorder kernel.

    This object will be used to create a PreBeamformReorder object that will be able to run the created kernel.

    The parameters given to this function are used by this class to compile the kernel and by the
    PreBeamformReorder to specify the shape of the memory buffers connected to this kernel.

    Parameters
    ----------
    context:
    The GPU device's context provided by katsdpsigproc's abstraction of PyCUDA.
    A context is associated with a single device and 'owns' all memory allocations.
    For the purposes of this python module the CUDA context is required.
    n_ants:
        The number of antennas that will be used in beamforming. Each antennas is expected to produce two polarisations.
    n_channels_per_stream:
        The number of frequency channels to be processed.
    n_samples_per_channel:
        The number of time samples to be processed per frequency channel.
    n_batches:
        The number of matrices to be reordered, a single data matrix = one batch.
    """

    def __init__(
        self,
        context: AbstractContext,
        n_ants: int,
        n_channels_per_stream: int,
        n_samples_per_channel: int,
        n_batches: int,
    ) -> None:
        """Initialise the PreBeamformReorderTemplate class and compile the pre-beamform reorder kernel."""
        # 1. Set member variables that are used to calculate indices for the input and output buffers
        self.n_ants = n_ants
        self.n_channels_per_stream = n_channels_per_stream
        self.n_samples_per_channel = n_samples_per_channel
        self.n_pols = 2  # Hardcoded to 2. No other values are supported
        self.n_batches = n_batches
        self._sample_bitwidth = 8
        self.complexity = 2

        # This 128 is hardcoded in the original tensor core kernel. The reason it is set to this needs to be determined.
        self.n_samples_per_block = 128 // self._sample_bitwidth
        self.n_blocks = self.n_samples_per_channel // self.n_samples_per_block

        if self.n_samples_per_channel % self.n_blocks != 0:
            raise ValueError(
                f"samples_per_channel must be divisible by {self.n_blocks}."
            )

        # 3. Declare the input and output data shapes
        self.inputDataShape = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(self.n_channels_per_stream, exact=True),
            accel.Dimension(self.n_samples_per_channel, exact=True),
            accel.Dimension(self.n_pols, exact=True),
            accel.Dimension(self.complexity, exact=True),
        )

        self.outputDataShape = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_pols, exact=True),
            accel.Dimension(self.n_channels_per_stream, exact=True),
            accel.Dimension(self.n_blocks, exact=True),
            accel.Dimension(self.n_samples_per_block, exact=True),
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(self.complexity, exact=True),
        )

        # The size of a data matrix required to be reordered is the same for Input or Output data shapes
        self.matrix_size = (
            self.n_ants
            * self.n_channels_per_stream
            * self.n_samples_per_channel
            * self.n_pols
        )

        # Maximum number of threads per block, as per Section I of Nvidia's CUDA Programming Guide
        threads_per_block: Final[int] = 1024
        self.threads_per_block = threads_per_block

        # 4. Calculate the number of thread blocks to launch per kernel call
        # - This is in the x-dimension and remains constant for the lifetime of the object.
        # - TODO: Error-check these values (As in, bounds/values, not method).
        self.n_blocks_x = int(np.ceil(self.matrix_size / threads_per_block))

        # 5. Compile the kernel
        #   - The size of this kernel simply depends on the individual matrix size and the
        #     number of batches required to be reordered.
        program = accel.build(
            context,
            "kernels/prebeamform_reorder_kernel.mako",
            {
                "n_ants": self.n_ants,
                "n_channels_per_stream": self.n_channels_per_stream,
                "n_samples_per_channel": self.n_samples_per_channel,
                "n_pols": self.n_pols,
                "n_samples_per_block": self.n_samples_per_block,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        self.kernel = program.get_kernel("prebeamform_reorder")

    def instantiate(
        self, command_queue: accel.AbstractCommandQueue
    ) -> "PreBeamformReorder":
        """Create a PreBeamformReorder object using this template to build the kernel."""
        return PreBeamformReorder(self, command_queue)


class PreBeamformReorder(accel.Operation):
    """
    Class containing a pre-beamform reorder kernel compiled from a PreBeamformReorderTemplate.

    .. rubric:: Slots
    inSamples: (n_batches, n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, complexity), uint8
        Input channelised data.
    outReordered: (n_batches, n_pols, n_channels_per_stream, n_blocks, n_samples_per_block, n_ants, complexity), uint8
        Output reordered data.

    This class specifies the shape of the input sample and output reordered buffers required by the kernel. The
    parameters specified in the PreBeamformReorderTemplate object are used to determine the shape of the buffers.

    It is worth noting these matrices follow the C convention, with the fastest-changing dimension being
    the last on the list.
    The input sample buffer must have the shape:
    [n_batches][antennas][n_channels_per_stream][n_samples_per_channel][n_pols][complexity]

    The output sample buffer must have the shape:
    [n_batches][n_pols][n_channels_per_stream][n_blocks][n_samples_per_block][n_ants][complexity]

    The n_samples_per_channel index is split over two different indices. The outer index ranges from 0 to n_blocks and
    the inner index from 0 to n_samples_per_channel//n_blocks (i.e n_samples_per_block). Times per block calculated by
    the PreBeamformReorderTemplate object.

    Each input element is a complex 8-bit integer sample.
    """

    def __init__(
        self,
        template: PreBeamformReorderTemplate,
        command_queue: accel.AbstractCommandQueue,
    ) -> None:
        """Initialise the PreBeamformReorder class."""
        super().__init__(command_queue)
        self.template = template
        self.slots["inSamples"] = accel.IOSlot(
            dimensions=self.template.inputDataShape, dtype=np.uint8
        )  # TODO: This must depend on input bitwidth
        self.slots["outReordered"] = accel.IOSlot(
            dimensions=self.template.outputDataShape, dtype=np.uint8
        )

    def _run(self) -> None:
        """Run the correlation kernel."""
        in_samples_buffer = self.buffer("inSamples")
        out_reordered_buffer = self.buffer("outReordered")
        max_thread_idx = int(self.template.threads_per_block * self.template.n_blocks_x)

        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [in_samples_buffer.buffer, out_reordered_buffer.buffer],
            # Even though we are using CUDA, we follow OpenCLs grid/block conventions. As such we need to multiply the
            # number of blocks(global_size) by the block size(local_size) in order to specify global threads not global
            # blocks.
            # - Global size is across the x- and y-dimensions (for this application).
            global_size=(max_thread_idx, self.template.n_batches),
            local_size=(self.template.threads_per_block, 1),
        )
