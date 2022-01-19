"""Coefficient generator for unit tests."""
import math

import numpy as np


class CoeffGenerator:
    """Class for generating coefficients for testing purposes.

    Note: The coefficients for this feature will be supplied by CAM and are not
    generated by CBF.

    Parameters
    ----------
    delay_vals: nd.array[float (single)]
        Data matrix of delay values.
    batches: int
        Number of batches to process.
    pols: int
        Number of polarisations.
    num_channels: int
        The number of channels the XEng core will process.
    total_channels: int
        The total number of channels in the system.
    n_blocks: int
        Number of blocks into which samples are divided in groups of 16
    samples_per_block: int
        Number of samples to process per sample-block
    n_ants: int
        The number of antennas from which data will be received.
    num_beams: int
         Number of beams to be steered.
    xeng_id: int
        Identify of the XEngine. This is used to compute the actual channel numbers processed per engine.
    sample_period: int
        Sampling period of the ADC.
    """

    def __init__(
        self,
        delay_vals,
        batches,
        num_pols,
        n_channels_per_stream,
        total_channels,
        n_blocks,
        samples_per_block,
        n_ants,
        num_beams,
        xeng_id,
        sample_period,
    ):
        """Initialise the coefficient generation class."""
        self.delay_vals = delay_vals
        self.batches = batches
        self.pols = num_pols
        self.num_channels = n_channels_per_stream
        self.total_channels = total_channels
        self.n_blocks = n_blocks
        self.samples_per_block = samples_per_block
        self.n_ants = n_ants
        self.num_beams = num_beams
        self.xeng_id = xeng_id
        self.sample_period = sample_period
        self.total_length = (
            self.batches
            * self.pols
            * self.num_channels
            * self.n_blocks
            * self.samples_per_block
        )
        self.complexity = 2  # Always

        # Static coefficient values for testing
        self.real_coeff_value = 4
        self.imag_coeff_value = 1

    def cpu_coeffs(self):
        """Generate coefficients for complex multiplication on the GPU.

        Note: This is for use in complex multiplication using two
        real-valued arrays. For this reason the coefficients need to be
        arranged as follows.

        Coefficients Array:
        [R00  I00
         -I00 R00
         R10  I10
         -I10 R10
         ...  ...]
        Where:  R00 = Real coeff 0
                I00 = Imag coeff 0
        and the matrix is structured as a N x 2 array.

        Returns
        -------
        coeffs: np.ndarray[np.float32].
            Output array of test coefficients.
        """
        cols = 2

        coeff_matrix = np.empty(
            self.batches
            * self.pols
            * self.num_channels
            * self.n_ants
            * self.num_beams
            * self.complexity
            * cols,
            dtype=np.float32,
        )
        coeff_matrix = coeff_matrix.reshape(
            self.batches,
            self.pols,
            self.num_channels,
            self.n_ants * self.complexity,
            self.num_beams * cols,
        )

        for ibatchindex in range(self.batches):
            for ipolindex in range(self.pols):
                for ichannelindex in range(self.num_channels):
                    for ibeamindex in range(self.num_beams):
                        for iantindex in range(self.n_ants):
                            delay_s = self.delay_vals[ichannelindex][ibeamindex][
                                iantindex
                            ][0]
                            # DelayRate_sps = self.delay_vals[ichannelindex][ibeamindex][iantindex][1]
                            phase_rad = self.delay_vals[ichannelindex][ibeamindex][
                                iantindex
                            ][2]
                            # PhaseRate_radps = self.delay_vals[ichannelindex][ibeamindex][iantindex][3]

                            # Compute actual channel index (i.e. channel in spectrum being computed on)
                            # This is needed when computing the rotation value before the cos/sin lookup.
                            # There are n_channels per xeng so adding n_channels * xeng_id gives the
                            # relative channel in the spectrum the xeng GPU thread is working on.
                            ichannel = ichannelindex + self.num_channels * self.xeng_id

                            # Part1:
                            # Take delay_CAM*channel_num_i*b-pi/(total_channels*sampling_rate_nanosec)+phase_offset_CAM
                            initial_phase = (
                                delay_s
                                * ichannel
                                * (-np.math.pi)
                                / (self.total_channels * self.sample_period)
                                + phase_rad
                            )

                            # Then: Compute phase correction atthe center of the band
                            # Part2:
                            # Compute as Delay_CAM * (total_channels/2) * -pi / (total_channels * sampling_rate_nanosec)
                            phase_correction_band_center = (
                                delay_s
                                * (self.total_channels / 2)
                                * (-np.math.pi)
                                / (self.total_channels * self.sample_period)
                            )

                            # Part3: Calculate rotation value
                            rotation = initial_phase - phase_correction_band_center

                            # Part4: Compute Steering Coeffs
                            steering_coeff_correct_real = math.cos(rotation)
                            steering_coeff_correct_imag = math.sin(rotation)

                            iantmatrix = iantindex * 2
                            ibeammatrix = ibeamindex * 2

                            # Part5: Store coeffs in return matrix
                            coeff_matrix[ibatchindex][ipolindex][ichannelindex][
                                iantmatrix
                            ][ibeammatrix + 1] = steering_coeff_correct_imag
                            coeff_matrix[ibatchindex][ipolindex][ichannelindex][
                                iantmatrix
                            ][ibeammatrix] = steering_coeff_correct_real

                            coeff_matrix[ibatchindex][ipolindex][ichannelindex][
                                iantmatrix + 1
                            ][ibeammatrix + 1] = steering_coeff_correct_real
                            coeff_matrix[ibatchindex][ipolindex][ichannelindex][
                                iantmatrix + 1
                            ][ibeammatrix] = -steering_coeff_correct_imag
        return coeff_matrix
