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

    for b in range(n_batches):
        for p in range(n_pols):
            for c in range(n_channels_per_stream):
                for block in range(n_blocks):
                    for s in range(n_samples_per_block):
                        d_cmplx = np.zeros(n_ants * 2).astype(np.float32)
                        c_cmplx = (
                            np.zeros(n_ants * 2 * 2)
                            .astype(np.float32)
                            .reshape(n_ants * 2, 2)
                        )

                        for beam in range(n_beams // 2):
                            for a in range(n_ants):
                                d_cmplx[a * 2] = input_data[b, p, c, block, s, a, 0]
                                d_cmplx[a * 2 + 1] = input_data[b, p, c, block, s, a, 1]

                                c_cmplx[a * 2, 0] = coeffs[b, p, c, a * 2, 0]
                                c_cmplx[a * 2, 0 + 1] = coeffs[b, p, c, a * 2, 0 + 1]

                                c_cmplx[a * 2 + 1, 0] = (
                                    -1 * coeffs[b, p, c, a * 2, 0 + 1]
                                )
                                c_cmplx[a * 2 + 1, 0 + 1] = coeffs[b, p, c, a * 2, 0]

                            # Compute
                            c_prod = np.dot(d_cmplx, c_cmplx)

                            # Assign real and imaginary results to repective positions
                            output_data[b, p, c, block, s, beam * 2] = c_prod[0]
                            output_data[b, p, c, block, s, beam * 2 + 1] = c_prod[1]


def complex_mult(
    input_data: np.ndarray,
    coeffs: np.ndarray,
    output_data_shape: tuple,
):
    """Compute complex multiplication on CPU for GPU verification.

    Parameters
    ----------
    input_data:
        Input data for reordering.
    output_data:
        Reordered data.
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
    n_ants:
        Number of antennas in array.
    n_beams:
        The number of beams that will be steered.

    Returns
    -------
    np.ndarray of type float
        Output array of beamformed data.
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
