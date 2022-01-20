"""
Unit test beamformer complex multiplication. This will be exucetd on the CPU.

The beamform multiplication kernel ingests data from the pre-beamform reorder and produces a beamformed product
as per the shape descibed.
"""
import numpy as np


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
    batches:
        Number of batches to process.
    pols:
        Numer of polarisations. Always 2.
    n_channel:
        Number of total channels per array.
    ants:
        Number of antennas in array.
    samples_chan:
        Number of samples per channels.
    n_samples_per_block:
        Number of samples per block.

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

    for b in range(n_batches):
        for p in range(n_pols):
            for c in range(n_channels_per_stream):
                for block in range(n_blocks):
                    for s in range(n_samples_per_block):
                        for beam in range(n_beams // 2):
                            data_cmplx = []
                            coeff_cmplx = []
                            for a in range(n_ants):
                                # Create complex valued pair for coefficients
                                dtmp_cmplx = complex(
                                    input_data[b, p, c, block, s, a, 0],
                                    input_data[b, p, c, block, s, a, 1],
                                )
                                # Append complex valued pair to form an array of <real,imag> values
                                data_cmplx.append(dtmp_cmplx)

                                # Create complex valued pair for coefficients
                                ctmp_cmplx = complex(
                                    coeffs[b, p, c, a * 2, beam * 2],
                                    coeffs[b, p, c, a * 2, beam * 2 + 1],
                                )

                                # Append complex valued pair to form an array of <real,imag> values
                                coeff_cmplx.append(ctmp_cmplx)

                            # Compute
                            cmplx_prod = np.dot(data_cmplx, coeff_cmplx)

                            # Assign real and imaginary results to repective positions
                            output_data[b, p, c, block, s, beam * 2] = np.real(
                                cmplx_prod
                            )
                            output_data[b, p, c, block, s, beam * 2 + 1] = np.imag(
                                cmplx_prod
                            )
    return output_data
