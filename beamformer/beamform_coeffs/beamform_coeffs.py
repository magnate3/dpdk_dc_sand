"""
Module for beamformer multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
Provision for batched operations is included, i.e. reordering multiple sets of data (matrices) passed to the kernel
in a single array.
"""
import numpy as np

# from beamforming.complex_mult_kernel import complex_mult_kernel
# from beamforming.cublas_SgemmBatched import cublas_SgemmBatched
from beamform_coeffs.beamformcoeff_kernel import BeamformCoeffKernel
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext
from katsdpsigproc.accel import IOSlot, Operation


class BeamformCoeffsTemplate:
    """
    Template class for beamform coeficient generator.

    Parameters
    ----------
    context:
        The GPU device's context provided by katsdpsigproc's abstraction of PyCUDA.
        A context is associated with a single device and 'owns' all memory allocations.
        For the purposes of this python module the CUDA context is required.
    delay_vals:
        Data matrix of delay values.
    n_beams:
        The number of beams that will be steered.
    n_ants:
        The number of antennas that will be used in beamforming. Each antennas is expected to produce two polarisations.
    n_channels:
        The number of frequency channels to be processed.
    """

    def __init__(
        self,
        context: AbstractContext,
        delay_vals: int,
        n_beams: int,
        n_ants: int,
        n_channels: int,
    ) -> None:
        self.context = context
        self.delay_vals = delay_vals
        self.n_beams = n_beams
        self.n_channels = n_channels
        self.n_ants = n_ants

        self.coeff_data_dimensions = (
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_beams, exact=True),
            accel.Dimension(self.n_ants, exact=True),
        )

    def instantiate(self, command_queue: accel.AbstractCommandQueue):
        """Initialise the complex multiplication class."""
        return CoeffGen(self, command_queue)


class CoeffGen(Operation):
    """Class for beamform complex multiplication.

    .. rubric:: Slots
    **inData** : (batches, n_pols, n_channels, n_blocks, n_samples_per_block, n_ants, complexity), uint8
        Input reordered channelised data.
    **outData** : (batches, n_pols, n_channels, n_blocks, n_samples_per_block, complexity), float32
        Beamformed data.
    **inCoeffs** : (batches, n_pols, n_channels, n_blocks, n_samples_per_block, complexity, n_ants, 2), float32
        Beamforming coefficients.

    Parameters
    ----------
    template: BeamformCoeffsTemplate
        Template for beamform coefficients class
    command_queue: accel.AbstractCommandQueue
        CUDA command queue
    """

    def __init__(self, template: BeamformCoeffsTemplate, command_queue: accel.AbstractCommandQueue):
        super().__init__(command_queue)
        self.template = template
        self.slots["OutCoeffs"] = IOSlot(dimensions=self.template.coeff_data_dimensions, dtype=np.float32)

    def _run(self):
        """Run the beamform computation."""
        with self.command_queue.context:
            BeamformCoeffKernel.coeff_gen(
                self, self.delay_vals, self.n_beams, self.n_channels, self.n_ants, self.buffer("OutCoeffs").buffer
            )
