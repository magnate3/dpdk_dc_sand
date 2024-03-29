"""Module for beamformer coefficient generation."""

import math

import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext
from katsdpsigproc.accel import IOSlot, Operation
from numba import cuda


@cuda.jit
def run_coeff_gen(
    delay_vals,
    n_batches,
    n_pols,
    n_channels_per_stream,
    n_channels,
    n_beams,
    n_ants,
    xeng_id,
    sample_period,
    coeffs,
):
    """Execute Beamforming steering coefficients."""
    # Compute flattened index inside the array
    ithreadindex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if ithreadindex_x < (n_batches * n_pols * n_channels_per_stream * n_beams * n_ants):
        # Compute indexes for delay_vals matrix
        ibatchindex_rem = ithreadindex_x % (
            n_pols * n_channels_per_stream * n_beams * n_ants
        )

        ipolindex_rem = ibatchindex_rem % (n_channels_per_stream * n_beams * n_ants)

        ichannelindex = ipolindex_rem // (n_beams * n_ants)
        ichannelindex_rem = ipolindex_rem % (n_beams * n_ants)

        iantindex = ichannelindex_rem // (n_beams)
        iantindex_rem = ichannelindex_rem % (n_beams)

        ibeamindex = iantindex_rem

        # Extract delay and phase values
        delay_s = delay_vals[ichannelindex][ibeamindex][iantindex][0]
        phase_rad = delay_vals[ichannelindex][ibeamindex][iantindex][2]

        # Compute actual channel index (i.e. channel in spectrum being computed on)
        # This is needed when computing the rotation value before the cos/sin lookup.
        # There are n_channels per xeng so adding n_channels * xeng_id gives the
        # relative channel in the spectrum the xeng GPU thread is working on.
        ichannel = ichannelindex + n_channels_per_stream * xeng_id

        initial_phase = (
            delay_s * ichannel * (-np.math.pi) / (n_channels * sample_period)
            + phase_rad
        )

        phase_correction_band_center = (
            delay_s * (n_channels / 2) * (-np.math.pi) / (n_channels * sample_period)
        )

        # Compute rotation value for steering coefficient computation
        rotation = initial_phase - phase_correction_band_center

        steering_coeff_correct_real = math.cos(rotation)
        steering_coeff_correct_imag = math.sin(rotation)

        # Compute indexes for output matrix
        ibatchindex = ithreadindex_x // (
            n_pols * n_channels_per_stream * n_beams * n_ants
        )
        ibatchindex_rem = ithreadindex_x % (
            n_pols * n_channels_per_stream * n_beams * n_ants
        )

        ipolindex = ibatchindex_rem // (n_channels_per_stream * n_beams * n_ants)
        ipolindex_rem = ibatchindex_rem % (n_channels_per_stream * n_beams * n_ants)

        ichannelindex = ipolindex_rem // (n_beams * n_ants)
        ichannelindex_rem = ipolindex_rem % (n_beams * n_ants)

        ibeamindex = ichannelindex_rem // (n_ants)
        ibeamindex_rem = ichannelindex_rem % (n_ants)

        ibeam_matrix = ibeamindex * 2
        iant_matrix = ibeamindex_rem * 2

        # Store steering coefficients in output matrix
        coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix][
            ibeam_matrix
        ] = steering_coeff_correct_real
        coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix][
            ibeam_matrix + 1
        ] = steering_coeff_correct_imag

        coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix + 1][
            ibeam_matrix
        ] = -steering_coeff_correct_imag
        coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix + 1][
            ibeam_matrix + 1
        ] = steering_coeff_correct_real


class CoeffGeneratorTemplate:
    """
    Template class for beamform coeficient generator.

    Parameters
    ----------
    context:
        The GPU device's context provided by katsdpsigproc's abstraction of PyCUDA.
        A context is associated with a single device and 'owns' all memory allocations.
        For the purposes of this python module the CUDA context is required.
    n_batches:
        Number of batches to process.
    n_pols:
        Number of polarisations.
    n_channels_per_stream:
        The number of channels the XEng core will process.
    n_channels:
        The total number of channels in the system.
    n_blocks:
        Number of blocks into which samples are divided in groups of 16
    n_samples_per_block:
        Number of samples to process per sample-block
    n_ants:
        The number of antennas from which data will be received.
    n_beams:
         Number of beams to be steered.
    xeng_id:
        Identify of the XEngine. This is used to compute the actual channel numbers processed per engine.
    sample_period:
        Sampling period of the ADC.
    """

    def __init__(
        self,
        context: AbstractContext,
        n_batches: int,
        n_pols: int,
        n_channels_per_stream: int,
        n_channels: int,
        n_blocks: int,
        n_samples_per_block: int,
        n_ants: int,
        n_beams: int,
        xeng_id: int,
        sample_period: float,
    ) -> None:
        self.context = context
        self.n_batches = n_batches
        self.n_pols = n_pols
        self.n_channels_per_stream = n_channels_per_stream
        self.n_channels = n_channels
        self.n_blocks = n_blocks
        self.n_samples_per_block = n_samples_per_block
        self.n_ants = n_ants
        self.n_beams = n_beams
        self.xeng_id = xeng_id
        self.sample_period = sample_period

        self.delay_vals_data_dimensions = (
            accel.Dimension(self.n_channels_per_stream, exact=True),
            accel.Dimension(self.n_beams, exact=True),
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(4, exact=True),
        )

        self.coeff_data_dimensions = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_pols, exact=True),
            accel.Dimension(self.n_channels_per_stream, exact=True),
            accel.Dimension(self.n_ants * 2, exact=True),
            accel.Dimension(self.n_beams * 2, exact=True),
        )

    def instantiate(self, command_queue: accel.AbstractCommandQueue):
        """Initialise the coefficient generation class."""
        return CoeffGenerator(self, command_queue)


class CoeffGenerator(Operation):
    """Class for beamform coefficient generation.

    Parameters
    ----------
    template: CoeffGeneratorTemplate
        Template for beamform coefficients class
    command_queue: accel.AbstractCommandQueue
        CUDA command queue
    """

    def __init__(
        self,
        template: CoeffGeneratorTemplate,
        command_queue: accel.AbstractCommandQueue,
    ):
        super().__init__(command_queue)
        self.template = template
        self.slots["delay_vals"] = IOSlot(
            dimensions=self.template.delay_vals_data_dimensions, dtype=np.float32
        )
        self.slots["outCoeffs"] = IOSlot(
            dimensions=self.template.coeff_data_dimensions, dtype=np.float32
        )

    def _run(self):
        """Run the coefficient generation."""
        threadsperblock = 128

        # Calculate the number of thread blocks in the grid
        blockspergrid = np.uint(
            np.ceil(
                self.template.n_batches
                * self.template.n_pols
                * self.template.n_channels
                * self.template.n_beams
                * self.template.n_ants
                / threadsperblock
            )
        )

        with self.command_queue.context:

            # Make the context associated with device device_id the current
            # context.
            # NOTE: Without doing this Numba will try execute kernel code on its
            # own context which will throw an error as the device already has a
            # context associated to it from katsdpsigproc command queue. This
            # will make the context associated with the deivce device_id the
            # current context.
            cuda.select_device(0)

            run_coeff_gen[blockspergrid, threadsperblock](
                self.buffer("delay_vals").buffer,
                self.template.n_batches,
                self.template.n_pols,
                self.template.n_channels_per_stream,
                self.template.n_channels,
                self.template.n_beams,
                self.template.n_ants,
                self.template.xeng_id,
                self.template.sample_period,
                self.buffer("outCoeffs").buffer,
            )

            # Wait for all commands in the stream to finish executing.
            cuda.synchronize()
