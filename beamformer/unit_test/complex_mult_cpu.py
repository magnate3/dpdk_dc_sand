"""
Unit test beamformer complex multiplication. This will be exucetd on the CPU.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
import numpy as np
from numba import njit


@njit
def run_cpu_cmplxmult(
    n_batches,
    n_pols,
    n_channels_per_stream,
    n_blocks,
    n_samples_per_block,
    n_beams,
    n_ants,
    input_data,
    coeffs,
    output_data,
):
    """Execute complex multiplication.

    Parameters
    ----------
    n_batches:
        Number of batches to process.
    n_pols:
        Numer of polarisations. Always 2.
    n_channels_per_stream:
        The number of channels the XEng core will process.
    n_blocks:
        Number of blocks into which samples are divided in groups of 16
    n_samples_per_block:
        Number of samples to process per sample-block
    n_beams:
        The number of beams that will be steered.
    n_ants:
        Number of antennas in array.
    input_data:
        Input data from reordering.
    coeffs:
        Coefficients used for beamforming.
    output_data:
        Beamformed data.

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

    The CPU version executes on one sample set of all antennas at a time. The set of antenna
    samples are multiplied with the coefficeint matrix and accumulated. Both the
    data and the coefficients used are complex valued requiring a complex multiplication.
    To utilise standard matrix mutliplication the coefficient matrix is constructed as detailed above.
    """
    for b in range(n_batches):
        for p in range(n_pols):
            for c in range(n_channels_per_stream):
                for block in range(n_blocks):
                    for s in range(n_samples_per_block):
                        data_cmplx = np.zeros(n_ants * 2).astype(np.float32)
                        coeff_cmplx = (
                            np.zeros(
                                n_ants * 2 * 2
                            )  # Note: The *2*2 is to accomodate the complex mult trick (above)
                            .astype(np.float32)
                            .reshape(n_ants * 2, 2)
                        )

                        for beam in range(n_beams // 2):
                            for a in range(n_ants):
                                data_cmplx[a * 2] = input_data[b, p, c, block, s, a, 0]
                                data_cmplx[a * 2 + 1] = input_data[
                                    b, p, c, block, s, a, 1
                                ]

                                coeff_cmplx[a * 2, 0] = coeffs[b, p, c, a * 2, 0]
                                coeff_cmplx[a * 2, 1] = coeffs[b, p, c, a * 2, 1]

                                coeff_cmplx[a * 2 + 1, 0] = (
                                    -1 * coeffs[b, p, c, a * 2, 1]
                                )
                                coeff_cmplx[a * 2 + 1, 1] = coeffs[b, p, c, a * 2, 0]

                            # Compute
                            cmplx_prod = np.dot(data_cmplx, coeff_cmplx)

                            # Assign real and imaginary results to repective positions
                            output_data[b, p, c, block, s, beam * 2] = cmplx_prod[0]
                            output_data[b, p, c, block, s, beam * 2 + 1] = cmplx_prod[1]


def complex_mult(
    input_data: np.ndarray,
    coeffs: np.ndarray,
    output_data_shape: tuple,
):
    """Compute complex multiplication on CPU for GPU verification.

    Parameters
    ----------
    input_data:
        input_data (reordered data).
    coeffs:
        Coefficients for beamforming computation.
    output_data_shape:
        Matrix dimensions which output data needs to match. .

    Returns
    -------
    output_data: np.ndarray of type float
        Complex multipication product for beamforming computation.
    """
    output_data = np.empty(output_data_shape, dtype=np.float32)
    n_batches = np.shape(input_data)[0]
    n_pols = np.shape(input_data)[1]
    n_channels_per_stream = np.shape(input_data)[2]
    n_blocks = np.shape(input_data)[3]
    n_samples_per_block = np.shape(input_data)[4]
    n_ants = np.shape(input_data)[5]
    n_beams = np.shape(coeffs)[4]

    run_cpu_cmplxmult(
        n_batches,
        n_pols,
        n_channels_per_stream,
        n_blocks,
        n_samples_per_block,
        n_beams,
        n_ants,
        input_data,
        coeffs,
        output_data,
    )
    return output_data
