"""Module containing correlator configuration options that should be tested where appropriate in all unit tests."""
import numpy as np

batches = [3]

# These are the estimated subarray sizes that will be run. The 130, 192 and 256 values are estimates for SKA and are
# not final. Additionally values 5,23,61 and 19 are just there to test that various non-power-of-two array sizes will
# run.
array_size = [4, 8, 16, 32, 64, 84, 130, 192, 256, 5, 23, 61, 19]

# array_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#              11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
#              21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
#              31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
#              41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#              51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
#              61, 62, 63, 64]

# This is always set to 256 for the MeerKAT case and leads to packet sizes equal to 256 * 4 bytes per sample = 1 KiB.
# There is a chance that for the MeerKAT extension this could be configured to allow for packet sizes equal to 2 or 4
# KiB. All these cases need to be tested.
num_samples_per_channel = [256, 512, 1024]

# Number of FFT channels out of the F-Engine
num_channels = [1024, 4096, 32768]

n_times_per_block = [16]