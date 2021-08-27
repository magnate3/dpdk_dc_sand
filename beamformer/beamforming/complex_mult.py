"""Reorder implementation for unit test."""
import numpy as np
from numba import jit


# @jit
def run_complex_mult(
    input_data: np.ndarray,
    output_data: np.ndarray,
    coeffs: np.ndarray,
    batches: int,
    pols: int,
    n_channel: int,
    blocks: int,
    n_samples_per_block: int,
    ants: int,
    complexity: int,
):
    """Reorder input data into provided datashape.

    Parameters
    ----------
    input_data: np.ndarray of type uint16
        Input data for reordering.
    output_data: np.ndarray of type uint16
        Reordered data.
    batches: int
        Number of batches to process.
    pols: int
        Numer of polarisations. Always 2.
    n_channel: int
        Number of total channels per array.
    ants: int
        Number of antennas in array.
    samples_chan: int
        Number of samples per channels.
    n_samples_per_block: int
        Number of samples per block.
    Returns
    -------
    np.ndarray of type uint16
        Output array of reshaped data.
    """

    # Option 1:
    for b in range(batches):
        for p in range(pols):
            for c in range(n_channel):
                for block in range(blocks):
                    for s in range(n_samples_per_block):
                        data_cmplx = []
                        coeff_cmplx = []
                        for a in range(ants):
                            dtmp_cmplx = complex(input_data[b, p, c, block, s, a, 0], input_data[b, p, c, block, s, a, 1])
                            data_cmplx.append(dtmp_cmplx)

                            ctmp_cmplx = complex(coeffs[b, p, c, block, s, a, 0], coeffs[b, p, c, block, s, a, 1])
                            coeff_cmplx.append(ctmp_cmplx)

                        cmplx_prod = np.vdot(data_cmplx,coeff_cmplx)
                        output_data[b, p, c, block, s, 1] = np.real(cmplx_prod)
                        output_data[b, p, c, block, s, 0] = np.imag(cmplx_prod)
                        # output_data[b, p, c, block, s] = input_data[b, p, c, s, p, cmplx]

    return output_data


def complex_mult(input_data: np.ndarray, coeffs: np.ndarray, output_data_shape: tuple) -> np.ndarray:
    """Reorder input data into provided datashape.

    Parameters
    ----------
    input_data: np.ndarray of type uint16
        Input data for reordering.
    input_data_shape: tuple
        Input data shape.
    output_data_shape: tuple
        Data shape to rehsape input data into.

    Returns
    -------
    np.ndarray of type uint16
        Output array of reshaped data.
    """
    output_data = np.empty(output_data_shape).astype(np.float32)

    batches = np.shape(input_data)[0]
    pols = np.shape(input_data)[1]
    n_channel = np.shape(input_data)[2]
    blocks = np.shape(input_data)[3]
    n_samples_per_block = np.shape(input_data)[4]
    ants = np.shape(input_data)[5]
    # complexity = np.shape(input_data)[6]

    run_complex_mult(input_data, output_data, coeffs, batches, pols, n_channel, blocks, n_samples_per_block, ants, 2)
    return output_data
