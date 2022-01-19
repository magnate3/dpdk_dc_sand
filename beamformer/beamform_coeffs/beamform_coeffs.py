"""
Module for beamformer multiplication.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
Provision for batched operations is included, i.e. reordering multiple sets of data (matrices) passed to the kernel
in a single array.
"""
import math

import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext
from katsdpsigproc.accel import IOSlot, Operation
from numba import cuda


@cuda.jit
def run_coeff_gen(
    delay_vals, batches, pols, n_channels, total_channels, n_beams, n_ants, xeng_id, sample_period, coeffs
):
    """Execute Beamforming steering coefficients."""
    # Compute flattened index inside the array
    ithreadindex_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if ithreadindex_x < (batches * pols * n_channels * n_beams * n_ants):
        # Compute indexes for delay_vals matrix
        ibatchindex_rem = ithreadindex_x % (pols * n_channels * n_beams * n_ants)

        ipolindex_rem = ibatchindex_rem % (n_channels * n_beams * n_ants)

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
        ichannel = ichannelindex + n_channels * xeng_id

        initial_phase = delay_s * ichannel * (-np.math.pi) / (total_channels * sample_period) + phase_rad

        phase_correction_band_center = delay_s * (total_channels / 2) * (-np.math.pi) / (total_channels * sample_period)

        # Compute rotation value for steering coefficient computation
        rotation = initial_phase - phase_correction_band_center

        steering_coeff_correct_real = math.cos(rotation)
        steering_coeff_correct_imag = math.sin(rotation)

        # Compute indexes for output matrix
        ibatchindex = ithreadindex_x // (pols * n_channels * n_beams * n_ants)
        ibatchindex_rem = ithreadindex_x % (pols * n_channels * n_beams * n_ants)

        ipolindex = ibatchindex_rem // (n_channels * n_beams * n_ants)
        ipolindex_rem = ibatchindex_rem % (n_channels * n_beams * n_ants)

        ichannelindex = ipolindex_rem // (n_beams * n_ants)
        ichannelindex_rem = ipolindex_rem % (n_beams * n_ants)

        ibeamindex = ichannelindex_rem // (n_ants)
        ibeamindex_rem = ichannelindex_rem % (n_ants)

        ibeam_matrix = ibeamindex * 2
        iant_matrix = ibeamindex_rem * 2

        # Store steering coefficients in output matrix
        coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix][ibeam_matrix] = steering_coeff_correct_real
        coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix][ibeam_matrix + 1] = steering_coeff_correct_imag

        coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix + 1][ibeam_matrix] = -steering_coeff_correct_imag
        coeffs[ibatchindex][ipolindex][ichannelindex][iant_matrix + 1][ibeam_matrix + 1] = steering_coeff_correct_real


class BeamformCoeffKernel:
    """Class for beamform coefficient kernel.

    Parameters
    ----------
    batches:
        Number of batches to process.
    num_pols:
        Number of polarisations.
    n_channels_per_stream:
        The number of channels the XEng core will process.
    total_channels:
        The total number of channels in the system.
    n_blocks:
        Number of blocks into which samples are divided in groups of 16
    samples_per_block:
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
        batches: int,
        num_pols: int,
        n_channels_per_stream: int,
        total_channels: int,
        n_blocks: int,
        samples_per_block: int,
        n_ants: int,
        n_beams: int,
        xeng_id: int,
        sample_period: int,
    ):
        self.batches = batches
        self.pols = num_pols
        self.n_channels = n_channels_per_stream
        self.total_channels = total_channels
        self.n_blocks = n_blocks
        self.samples_per_block = samples_per_block
        self.n_ants = n_ants
        self.n_beams = n_beams
        self.xeng_id = xeng_id
        self.sample_period = sample_period
        self.total_length = self.batches * self.pols * self.n_channels * self.n_blocks * self.samples_per_block
        self.complexity = 2  # Always

    def coeff_gen(self, delay_vals, coeff_matrix):
        """Complex multiplication setup."""
        threadsperblock = 128

        # Calculate the number of thread blocks in the grid
        blockspergrid = np.uint(
            np.ceil(self.batches * self.pols * self.n_channels * self.n_beams * self.n_ants / threadsperblock)
        )

        # Make the context associated with device device_id the current context.
        # NOTE: Without doing this Numba will try execute kernel code on it's own context which will throw an error as
        # the device already has a context associated to it from katsdpsigproc command queue. This will make the
        # context associated with the deivce device_id the current context.
        cuda.select_device(0)

        # Now start the kernel
        run_coeff_gen[blockspergrid, threadsperblock](
            delay_vals,
            self.batches,
            self.pols,
            self.n_channels,
            self.total_channels,
            self.n_beams,
            self.n_ants,
            self.xeng_id,
            self.sample_period,
            coeff_matrix,
        )

        # Wait for all commands in the stream to finish executing.
        cuda.synchronize()


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
        batches: int,
        num_pols: int,
        n_channels_per_stream: int,
        n_channels: int,
        n_blocks: int,
        samples_per_block: int,
        n_ants: int,
        n_beams: int,
        xeng_id: int,
        sample_period: float,
    ) -> None:
        self.context = context
        self.batches = batches
        self.num_pols = num_pols
        self.n_channels_per_stream = n_channels_per_stream
        self.n_channels = n_channels
        self.n_blocks = n_blocks
        self.samples_per_block = samples_per_block
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
            accel.Dimension(self.batches, exact=True),
            accel.Dimension(self.num_pols, exact=True),
            accel.Dimension(self.n_channels_per_stream, exact=True),
            accel.Dimension(self.n_ants * 2, exact=True),
            accel.Dimension(self.n_beams * 2, exact=True),
        )

    def instantiate(self, command_queue: accel.AbstractCommandQueue):
        """Initialise the complex multiplication class."""
        return BeamformCoeffs(self, command_queue)


class BeamformCoeffs(Operation):
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

        self.beamformcoeffkernel = BeamformCoeffKernel(
            self.template.batches,
            self.template.num_pols,
            self.template.n_channels_per_stream,
            self.template.n_channels,
            self.template.n_blocks,
            self.template.samples_per_block,
            self.template.n_ants,
            self.template.n_beams,
            self.template.xeng_id,
            self.template.sample_period,
        )
        self.slots["delay_vals"] = IOSlot(dimensions=self.template.delay_vals_data_dimensions, dtype=np.float32)
        self.slots["outCoeffs"] = IOSlot(dimensions=self.template.coeff_data_dimensions, dtype=np.float32)

    def _run(self):
        """Run the beamform computation."""
        with self.command_queue.context:
            self.beamformcoeffkernel.coeff_gen(self.buffer("delay_vals").buffer, self.buffer("outCoeffs").buffer)
