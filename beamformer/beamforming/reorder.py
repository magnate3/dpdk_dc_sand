"""Reorder implementation for unit test."""
import numpy as np
from numba import njit


@njit
def run_reorder(
    input_data: np.ndarray,
    output_data: np.ndarray,
    n_batches: int,
    n_pols: int,
    n_channel: int,
    n_ants: int,
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
    n_batches: int
        Number of batches to process.
    n_pols: int
        Numer of polarisations. Always 2.
    n_channel: int
        Number of total channels per array.
    n_ants: int
        Number of antennas in array.
    n_samples_per_block: int
        Number of samples per block.
    Returns
    -------
    np.ndarray of type uint16
        Output array of reshaped data.
    """
    output_data[:] = input_data.reshape(
        n_batches, n_ants, n_channel, -1, n_samples_per_block, n_pols, complexity
    ).transpose(0, 5, 2, 3, 4, 1, 6)
    return output_data


def reorder(
    input_data: np.ndarray, input_data_shape: tuple, output_data_shape: tuple
) -> np.ndarray:
    """Reorder input data into provided datashape.

    Parameters
    ----------
    input_data:
        Input data for reordering.
    input_data_shape:
        Input data shape.
    output_data_shape:
        Data shape to rehsape input data into.

    Returns
    -------
    np.ndarray of type uint16
        Output array of reshaped data.
    """
    output_data = np.empty(output_data_shape).astype(np.uint8)

    n_batches = input_data_shape[0]
    n_ants = input_data_shape[1]
    n_channel = input_data_shape[2]
    n_pols = input_data_shape[4]
    n_samples_per_block = output_data_shape[4]
    complexity = input_data_shape[5]

    run_reorder(
        input_data,
        output_data,
        n_batches,
        n_pols,
        n_channel,
        n_ants,
        n_samples_per_block,
        complexity,
    )
    return output_data
