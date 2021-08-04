"""Reorder implementation for unit test."""
import numpy as np
from numba import njit


@njit
def run_reorder(
    input_data: np.ndarray,
    output_data: np.ndarray,
    batches: int,
    pols: int,
    n_channel: int,
    ants: int,
    samples_chan: int,
    n_samples_per_block: int,
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
    # for b in range(batches):
    #     for p in range(pols):
    #         for c in range(n_channel):
    #             for a in range(ants):
    #                 for s in range(samples_chan):
    #                     for cmplx in range(complexity):
    #                         to = int(s / n_samples_per_block) # to = timeOuter
    #                         ti = int(s % n_samples_per_block) # ti = timeInner
    #                         output_data[b, p, c, to, ti, a, cmplx] = input_data[b, a, c, s, p, cmplx]

    # or

    # Option 2:
    output_data[:] = input_data.reshape(batches, ants, n_channel, -1, n_samples_per_block, pols, complexity).transpose(
        0, 5, 2, 3, 4, 1, 6
    )
    return output_data


def reorder(input_data: np.ndarray, input_data_shape: tuple, output_data_shape: tuple) -> np.ndarray:
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
    output_data = np.empty(output_data_shape).astype(np.uint8)

    batches = input_data_shape[0]
    ants = input_data_shape[1]
    n_channel = input_data_shape[2]
    samples_chan = input_data_shape[3]
    pols = input_data_shape[4]
    n_samples_per_block = output_data_shape[4]
    complexity = input_data_shape[5]

    run_reorder(input_data, output_data, batches, pols, n_channel, ants, samples_chan, n_samples_per_block, complexity)
    return output_data
