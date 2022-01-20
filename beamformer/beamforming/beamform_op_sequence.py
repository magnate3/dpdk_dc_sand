"""
Module wrapping the pre-beamform reorder Kernel and beamformer multiplication.

The pre-beamform reorder kernel operates on a set of data with dimensions explained (and in its _kernel.mako file).
The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product as
per the shape descibed. Provision for batched operations is included, i.e. reordering multiple sets of data (matrices)
passed to the kernel in a single array.
"""
from beamforming.coeff_generator import CoeffGeneratorTemplate
from beamforming.matrix_multiply import MatrixMultiplyTemplate
from beamforming.prebeamform_reorder import PreBeamformReorderTemplate
from katsdpsigproc import accel


class OpSequenceTemplate:
    """
    Template class for compiling beamform reorder, coefficient generation and beamform multiplication.

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
    n_batches: int
        The number of matrices to be reordered, a single data matrix = one batch.
    n_pols:
        Number of polarisations.
    n_channels_per_stream:
        The number of channels the XEng core will process.
    n_channels: int
        The total number of frequency channels out of the FFT.
    n_blocks:
        Number of blocks into which samples are divided in groups of 16
    samples_per_block:
        Number of samples to process per sample-block
    n_ants: int
        The number of antennas that will be used in beamforming. Each antennas is expected to produce two polarisations.
    n_beams:
        The number of beams that will be steered.
    xeng_id:
        Identify of the XEngine. This is used to compute the actual channel numbers processed per engine.
    sample_period:
        Sampling period of the ADC.
    n_samples_per_channel: int
        The number of time samples to be processed per channel.
    """

    def __init__(
        self,
        context,
        n_batches,
        n_pols,
        n_channels_per_stream,
        n_channels,
        n_blocks,
        samples_per_block,
        n_ants,
        n_beams,
        xeng_id,
        sample_period,
        n_samples_per_channel,
    ) -> None:
        """Initialise the BeamformSeqTemplate class."""

        self.preBeamformReorder_template = PreBeamformReorderTemplate(
            context, n_ants, n_channels_per_stream, n_samples_per_channel, n_batches
        )

        self.beamform_coeff_template = CoeffGeneratorTemplate(
            context,
            n_batches,
            n_pols,
            n_channels_per_stream,
            n_channels,
            n_blocks,
            samples_per_block,
            n_ants,
            n_beams,
            xeng_id,
            sample_period,
        )

        self.beamform_mult_template = MatrixMultiplyTemplate(
            context=context,
            n_ants=n_ants,
            n_channels_per_stream=n_channels_per_stream,
            n_samples_per_channel=n_samples_per_channel,
            n_beams=n_beams,
            n_batches=n_batches,
        )

    def instantiate(self, queue):
        """Instantiate and return OpSequence object."""
        return OpSequence(self, queue)


class OpSequence(accel.OperationSequence):
    """
    Class for OpSequence. This will link the following operations:
    1. pre-beamform reorder.
    2. beamform coeff generator.
    3. beamforming.

    Parameters
    ----------
    template: PostprocTemplate
        The template for the post-processing operation.
    command_queue: AbstractCommandQueue
        The GPU command queue (typically this will be a CUDA Stream) on which
        actual processing operations are to be scheduled.
    """

    def __init__(self, template, queue):
        """Initialise the OpSequence class."""
        self.prebeamform_reorder = template.preBeamformReorder_template.instantiate(
            queue
        )
        self.beamform_coeff = template.beamform_coeff_template.instantiate(queue)
        self.beamform_mult = template.beamform_mult_template.instantiate(queue)

        operations = [
            ("prebeamform_reorder", self.prebeamform_reorder),
            ("beamform_coeff", self.beamform_coeff),
            ("beamform_mult", self.beamform_mult),
        ]

        compounds = {
            "bufin_delay_vals": ["beamform_coeff:delay_vals"],
            "bufint_coeff": ["beamform_coeff:outCoeffs", "beamform_mult:inCoeffs"],
            "bufin_reorder": ["prebeamform_reorder:inSamples"],
            "bufint_data": ["prebeamform_reorder:outReordered", "beamform_mult:inData"],
            "bufout_mult": ["beamform_mult:outData"],
        }

        super().__init__(queue, operations, compounds)
        self.template = template
