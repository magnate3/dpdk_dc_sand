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
    input_data: np.ndarray[np.uint16]
        Input data for reordering.
    output_data: np.ndarray[np.uint16]
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
    np.ndarray of type float
        Output array of beamformed data.
    """
    output_data = np.empty(output_data_shape, dtype=np.float32)
    batches = np.shape(input_data)[0]
    pols = np.shape(input_data)[1]
    n_channel = np.shape(input_data)[2]
    blocks = np.shape(input_data)[3]
    n_samples_per_block = np.shape(input_data)[4]
    ants = np.shape(input_data)[5]
    beams = np.shape(coeffs)[4]

    for b in range(batches):
        for p in range(pols):
            for c in range(n_channel):
                if c == 1:
                    tt = 1
                for block in range(blocks):
                    for s in range(n_samples_per_block):
                        # data_cmplx = []
                        # coeff_cmplx = []
                        for beam in range(beams//2):
                            data_cmplx = []
                            coeff_cmplx = []
                            # print('beam:', beam)
                            for a in range(ants):
                                # Create complex valued pair for coefficients
                                dtmp_cmplx = complex(
                                    input_data[b, p, c, block, s, a, 0], input_data[b, p, c, block, s, a, 1]
                                )
                                # Append complex valued pair to form an array of <real,imag> values
                                data_cmplx.append(dtmp_cmplx)

                                # Create complex valued pair for coefficients
                                # ctmp_cmplx = complex(coeffs[b, p, c, block, s, a, 0], coeffs[b, p, c, block, s, a, 1])
                                ctmp_cmplx = complex(coeffs[b, p, c, a*2, beam*2], coeffs[b, p, c, a*2, beam*2+1])
                                # print('Ant:', ctmp_cmplx)
                                # Append complex valued pair to form an array of <real,imag> values
                                coeff_cmplx.append(ctmp_cmplx)

                            # Compute
                            cmplx_prod = np.dot(data_cmplx, coeff_cmplx)

                            # if (b == 0) & (p == 0) & (c == 1) & (block == 0) & (s == 0):
                            # # if (b == 0) & (p == 0) & (c == 1):
                            #     print('data_cmplx', data_cmplx)
                            #     print('beam:', (beam*2 + 1), 'coeff_cmplx', coeff_cmplx)
                            #     print('cmplx_prod',cmplx_prod)

                            # Assign real and imaginary results to repective positions
                            output_data[b, p, c, block, s, beam*2] = np.real(cmplx_prod)
                            output_data[b, p, c, block, s, beam*2 + 1] = np.imag(cmplx_prod)
    return output_data
