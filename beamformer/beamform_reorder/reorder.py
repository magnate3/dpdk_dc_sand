"""Reorder implementation for unit test."""
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from numba import njit, jit

@njit
def run_reorder(input_data, output_data, output_data_shape, batches, pols, n_channel, ants, samples_chan, n_times_per_block):
    #reorder_data = np.empty(output_data_shape).astype(np.int8)

    for batch in range(batches):
        for pol in range(pols):
            for chan in range(n_channel):
                for ant in range(ants):
                    for sample in range(samples_chan):
                        timeOuter = int(sample/n_times_per_block)
                        timeInner = int(sample % n_times_per_block)
                        output_data[batch][pol][chan][timeOuter][timeInner,ant] = input_data[batch][ant][chan][sample,pol]
    return output_data

def reorder(input_data: float, input_data_shape: tuple, output_data_shape: tuple) -> np.ndarray:
    """ Reorder input data into provided datashape.

    Parameters
    ----------
    input_data: float
        Input data for reordering.
    input_data_shape: tuple
        Input data shape.
    output_data_shape: tuple
        Data shape to rehsape input data into.

    Returns
    -------
    np.ndarray of type float
        Output array of reshaped data.
    """
    # output_data = np.empty(output_data_shape).astype(np.complex)
    # output_data = np.empty(output_data_shape).astype(np.uint8)
    output_data = np.empty(output_data_shape).astype(np.uint16)

    batches = input_data_shape[0]
    ants = input_data_shape[1]
    n_channel = input_data_shape[2]
    samples_chan = input_data_shape[3]
    pols = input_data_shape[4]
    n_times_per_block = output_data_shape[4]

    run_reorder(input_data, output_data, output_data_shape, batches, pols, n_channel, ants, samples_chan, n_times_per_block)
    return output_data

    # for batch in range(batches):
    #     for pol in range(pols):
    #         for chan in range(n_channel):
    #             for ant in range(ants):
    #                 for sample in range(samples_chan):
    #                     timeOuter = int(sample/n_times_per_block)
    #                     timeInner = int(sample % n_times_per_block)
    #                     # output_data[batch][pol][chan][:,ant] = input_data[batch][ant][chan][:,pol]
    #                     #test = input_data[batch][ant][chan][sample,pol]
    #                     output_data[batch][pol][chan][timeOuter][timeInner,ant] = input_data[batch][ant][chan][sample,pol]
    #                     #print(input_data[batch][ant][chan][sample,pol])

    #return output_data

# print("{}".format(config.ants))
# print(config.Dataset2[0][0][0][:,0])

# plt.figure(1)
# plt.plot(config.Dataset1[0][0][0][:,1])
# plt.show()
