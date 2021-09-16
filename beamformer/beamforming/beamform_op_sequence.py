"""
Module wrapping the pre-beamform reorder Kernel and beamformer multiplication.

The pre-beamform reorder kernel operates on a set of data with dimensions explained (and in its _kernel.mako file).
The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product as
per the shape descibed. Provision for batched operations is included, i.e. reordering multiple sets of data (matrices)
passed to the kernel in a single array.
"""

from beamform_reorder import prebeamform_reorder
from beamforming import matrix_multiply
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext
from unit_test import coeff_generator


class BeamformSeqTemplate:
    """
    Template class for compiling beamform reorder and beamform multiplication.

    This class specifies the shape of the input sample and output beamformed data.
    The parameters specified in the PreBeamformReorderTemplate object and beamformMult object
    are used to determine the shape of the buffers.

    It is worth noting these matrices follow the C convention, with the fastest-changing dimension being
    the last on the list.
    The input sample buffer must have the shape:
    [batch][antennas][channels][samples_per_channel][polarisations][complexity]

    The output beamforming buffer must have the shape:
    [n_batches][polarizations][n_channels][n_blocks][samples_per_block][complexity]

    The samples_per_channel index is split over two different indices. The outer index ranges from 0 to n_blocks and
    the inner index from 0 to samples_per_channel//n_blocks (i.e sample_per_block). Times per block is calculated by
    the PreBeamformReorderTemplate object.

    Each input element is a complex 8-bit integer sample.
    Each output sample is a complex 32b float.

    Parameters
    ----------
    context: cuda.Context
        The GPU device's context provided by katsdpsigproc's abstraction of PyCUDA.
        A context is associated with a single device and 'owns' all memory allocations.
        For the purposes of this python module the CUDA context is required.
    n_ants: int
        The number of antennas that will be used in beamforming. Each antennas is expected to produce two polarisations.
    n_channels: int
        The number of frequency channels to be processed.
    n_samples_per_channel: int
        The number of time samples to be processed per channel.
    n_batches: int
        The number of matrices to be reordered, a single data matrix = one batch.
    """

    def __init__(
        self,
        context: AbstractContext,
        n_ants: int,
        n_channels: int,
        n_samples_per_channel: int,
        n_batches: int,
        test_id,
    ) -> None:
        """Initialise the BeamformSeqTemplate class."""
        self.preBeamformReorder = prebeamform_reorder.PreBeamformReorderTemplate(
            context, n_ants, n_channels, n_samples_per_channel, n_batches
        )
        self.beamformMult = matrix_multiply.MatrixMultiplyTemplate(
            context, n_ants, n_channels, n_samples_per_channel, n_batches, test_id
        )

    def instantiate(self, queue, test_id):
        """Instantiate and return OpSequence object."""
        return OpSequence(self, queue, test_id)


class OpSequence(accel.OperationSequence):
    """
    Class for OpSequence. This will link the two operations of pre-beamform reorder.

    Parameters
    ----------
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

    def __init__(self, template, queue, test_id):
        """Initialise the OpSequence class."""
        self.prebeamformReorder = template.preBeamformReorder.instantiate(queue)
        self.beamformMult = template.beamformMult.instantiate(queue, test_id)
        operations = [("reorder", self.prebeamformReorder), ("beamformMult", self.beamformMult)]
        compounds = {
            "coeff_bufin": ["beamformMult:inCoeffs"],
            "bufin": ["reorder:inSamples"],
            "bufint": ["reorder:outReordered", "beamformMult:inData"],
            "bufout": ["beamformMult:outData"],
        }
        super().__init__(queue, operations, compounds)
        self.template = template


def print_debug(host_out):
    """Debug: Print out all the entries to verify values."""
    # TODO: Remove
    for b in range(host_out.shape[0]):
        for p in range(host_out.shape[1]):
            for c in range(host_out.shape[2]):
                for bl in range(host_out.shape[3]):
                    for s in range(host_out.shape[4]):
                        for cmplx in range(host_out.shape[5]):
                            print(host_out[b][p][c][bl][s][cmplx])


if __name__ == "__main__":
    # Reorder Specs
    batches = 3
    n_ants = 4
    num_channels = 1024
    num_samples_per_channel = 256
    pols = 2
    n_channels_per_stream = num_channels // n_ants // 4
    samples_per_block = 16
    n_blocks = num_samples_per_channel // samples_per_block

    # NOTE: test_id is a temporary inclusion meant to identify which complex multiply to call.
    # Options:  'sgemm' for cublas matrix mult
    #           'kernel' for numba-based complex multiplication kernel
    test_id = "kernel"

    # Generate coefficients
    coeff_gen = coeff_generator.CoeffGenerator(batches, n_channels_per_stream, n_blocks, samples_per_block, n_ants)
    if test_id == "kernel":
        coeffs = coeff_gen.GPU_Coeffs_kernel()
    elif test_id == "sgemm":
        coeffs = coeff_gen.GPU_Coeffs_cublas

    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda, interactive=False)
    queue = ctx.create_command_queue()
    op_template = BeamformSeqTemplate(ctx, n_ants, n_channels_per_stream, num_samples_per_channel, batches, test_id)
    op = op_template.instantiate(queue, test_id)
    op.ensure_all_bound()

    bufcoeff_device = op.beamformMult.buffer("inCoeffs")
    host_coeff = bufcoeff_device.empty_like()
    host_coeff = coeffs

    bufin_device = op.prebeamformReorder.buffer("inSamples")
    host_in = bufin_device.empty_like()

    bufout_device = op.beamformMult.buffer("outData")
    host_out = bufout_device.empty_like()

    # --- Inject ones data for test ---
    host_in[:] = 1
    # Or

    # --- Inject random data for test ---
    # rng = np.random.default_rng(seed=2021)
    # host_in[:] = rng.uniform(
    #     np.iinfo(host_in.dtype).min, np.iinfo(host_in.dtype).max, host_in.shape
    # ).astype(host_in.dtype)

    bufcoeff_device.set(queue, host_coeff)
    bufin_device.set(queue, host_in)
    op()
    bufout_device.get(queue, host_out)

    # Debug: Print out all the entries to verify values
    print_debug(host_out)

    # Visualise the operation (Just for interest)
    accel.visualize_operation(op, "test_op_vis")
