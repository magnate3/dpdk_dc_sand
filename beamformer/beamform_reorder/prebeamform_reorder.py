"""
Module wrapping the pre-correlation reorder Kernel.

The pre-correlation reorder kernel operates on a set of data with dimensions explained below (and in its _kernel.mako file).
It makes provision for batched operations, i.e. reordering multiple sets of data (matrices) passed to the kernel in a single array.

This module has two classes:
    1. PreBeamformReorderTemplate
        - This class allows for multiple different compilations of the same kernel with different parameters to take place.
    2. PreBeamformReorder
        - This class provides the interface to call the kernel created in a PreBeamformReorderTemplate object.

TODO:
    1. Update naming conventions as necessary.

"""

import pkg_resources
import numpy as np
from katsdpsigproc import accel
from katsdpsigproc import cuda
from typing_extensions import Final


class PreBeamformReorderTemplate:
    """
    Template class for compiling different variations of the pre-beamform reorder kernel.

    This object will be used to create a PreBeamformReorder object that will be able to run the created kernel.
    """

    def __init__(
        self, context: cuda.Context, n_ants: int, n_channels: int, n_samples_per_channel: int, n_batches: int
    ) -> None:
        """
        Initialise the PreBeamformReorderTemplate class and compile the pre-beamform reorder kernel.

        The parameters given to this function are used by this class to compile the kernel and by the
        PreBeamformReorder to specify the shape of the memory buffers connected to this kernel.

        Parameters
        ----------
        context: cuda.Context
            The GPU device's context provided by katsdpsigproc's abstraction of PyCUDA.
            A context is associated with a single device and 'owns' all memory allocations.
            For the purposes of this python module, and its Tensor Core usage, the CUDA context is required.
        n_ants: int
            The number of antennas that will be used in beamforming. Each antennas is expected to produce two polarisations.
        n_channels: int
            The number of frequency channels to be processed.
        n_samples_per_channel: int
            The number of time samples to be processed per frequency channel.
        n_batches: int
            The number of matrices to be reordered, a single data matrix = one batch.
        """
        # 1. Set member variables that are used to calculate indices for the input and output buffers
        self.n_ants = n_ants
        self.n_channels = n_channels
        self.n_samples_per_channel = n_samples_per_channel
        self.n_polarisations = 2  # Hardcoded to 2. No other values are supported
        self.n_batches = n_batches

        # This is set to 8 for now, but must be updated to 4- and 16-bit
        # as and when the TensorCoreXEngine requires it.
        self._sample_bitwidth = 8

        # This 128 is hardcoded in the original Tensor Core kernel and (probably) has to do with
        # optimising the thread utilisation in Tensor Cores - 128 = 4 x warps, where one warp = 32 threads.
        self.n_times_per_block = 128 // self._sample_bitwidth

        if self.n_samples_per_channel % self.n_times_per_block != 0:
            raise ValueError(f"samples_per_channel must be divisible by {self.n_times_per_block}.")

        # 3. Declare the input and output data shapes
        self.inputDataShape = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_samples_per_channel, exact=True),
            accel.Dimension(self.n_polarisations, exact=True),
        )

        self.outputDataShape = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_polarisations, exact=True),
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_samples_per_channel // self.n_times_per_block, exact=True),
            accel.Dimension(self.n_times_per_block, exact=True),
            accel.Dimension(self.n_ants, exact=True),
        )

        # self.outputDataShape = (
        #     self.n_batches,
        #     self.n_channels,
        #     self.n_samples_per_channel // self.n_times_per_block,
        #     self.n_ants,
        #     self.n_polarisations,
        #     self.n_times_per_block,
        # )

        # The size of a data matrix required to be reordered is the same for Input or Output data shapes
        self.matrix_size = self.n_ants * self.n_channels * self.n_samples_per_channel * self.n_polarisations
        # Maximum number of threads per block, as per Section I of Nvidia's CUDA Programming Guide
        THREADS_PER_BLOCK: Final[int] = 1024

        # 4. Calculate the number of thread blocks to launch per kernel call
        # - This is in the x-dimension and remains constant for the lifetime of the object.
        # - TODO: Error-check these values (As in, bounds/values, not method).
        # self.n_blocks_x = (self.matrix_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        # self.n_blocks_x = ((self.matrix_size + THREADS_PER_BLOCK) // THREADS_PER_BLOCK) - 1


        #self.n_blocks_x = self.matrix_size // THREADS_PER_BLOCK
        self.n_blocks_x = int(np.ceil(self.matrix_size / THREADS_PER_BLOCK))
        #self.n_blocks_x = float(self.matrix_size / THREADS_PER_BLOCK)

        # 5. Compile the kernel
        #   - The size of this kernel simply depends on the individual matrix size and the
        #     number of batches required to be reordered.
        program = accel.build(
            context,
            "kernels/prebeamform_reorder_kernel.mako",
            {
                "n_ants": self.n_ants,
                "n_channels": self.n_channels,
                "n_samples_per_channel": self.n_samples_per_channel,
                "n_polarisations": self.n_polarisations,
                "n_times_per_block": self.n_times_per_block,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        self.kernel = program.get_kernel("prebeamform_reorder")

    def instantiate(self, command_queue: accel.AbstractCommandQueue) -> "PreBeamformReorder":
        """Create a PreBeamformReorder object using this template to build the kernel."""
        return PreBeamformReorder(self, command_queue)


class PreBeamformReorder(accel.Operation):
    """
    Class containing a pre-correlation reorder kernel compiled from a PreBeamformReorderTemplate.

    This class specifies the shape of the input sample and output reordered buffers required by the kernel. The
    parameters specified in the PreBeamformReorderTemplate object are used to determine the shape of the buffers.

    It is worth noting these matrices follow the C convention, with the fastest-changing dimension being the last on the list.
    The input sample buffer must have the shape:
    [batch][antennas][channels][samples_per_channel][polarisations]

    The output sample buffer must have the shape:
    [batch][channels][samples_per_channel//times_per_block][n_ants][polarisations][times_per_block]

    A complexity that is introduced by the pre-correlation reorder kernel is that the samples_per_channel index is split over two
    different indices. The first index ranges from 0 to samples_per_channel//times_per_block and the second index
    ranges from 0 to times_per_block. Times per block is calculated by the PreBeamformReorderTemplate object.
    In 8-bit input mode times_per_block is equal to 16.

    Each input element is a complex 8-bit integer sample. Numpy does not support 8-bit complex numbers,
    so the input sample array has dtype of np.int16 as a placeholder.
    """

    def __init__(self, template: PreBeamformReorderTemplate, command_queue: accel.AbstractCommandQueue) -> None:
        """Initialise the PreBeamformReorder object and specify the size of the memory buffers."""
        super().__init__(command_queue)
        self.template = template
        self.slots["inSamples"] = accel.IOSlot(
            dimensions=self.template.inputDataShape, dtype=np.uint16,
        )  # TODO: This must depend on input bitwidth
        self.slots["outReordered"] = accel.IOSlot(dimensions=self.template.outputDataShape, dtype=np.uint16)

    def _run(self) -> None:
        """Run the correlation kernel."""
        inSamples_buffer = self.buffer("inSamples")
        outReordered_buffer = self.buffer("outReordered")
        # outTest_buffer = self.buffer("outTest")
        max_threadIdx = int(1024 * self.template.n_blocks_x)
        # max_threadIdx = 1024 * self.template.n_blocks_x

        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [inSamples_buffer.buffer, outReordered_buffer.buffer],
            # Even though we are using CUDA, we follow OpenCLs grid/block conventions. As such we need to multiply the number
            # of blocks(global_size) by the block size(local_size) in order to specify global threads not global blocks.
            # - Global size is across the x- and y-dimensions (for this application).
            global_size=(max_threadIdx, self.template.n_batches),
            local_size=(1024, 1),
        )
        # self.command_queue.enqueue_kernel(
        #     self.template.kernel,
        #     [inSamples_buffer.buffer, outReordered_buffer.buffer],
        #     # Even though we are using CUDA, we follow OpenCLs grid/block conventions. As such we need to multiply the number
        #     # of blocks(global_size) by the block size(local_size) in order to specify global threads not global blocks.
        #     # - Global size is across the x- and y-dimensions (for this application).
        #     global_size=(1024 * self.template.n_blocks_x, self.template.n_batches),
        #     local_size=(1024, 1),
        # )