"""Module containing correlator configuration options that should be tested where appropriate in all unit tests."""
import numpy as np

# """batches: Number of batches for processing."""
n_batches = [3]

"""n_ants: List of array sizes."""
# These are the estimated subarray sizes that will be run. The 130, 192 and 256 values are estimates for SKA and are
# not final. Additionally values 5,23,61 and 19 are just there to test that various non-power-of-two array sizes will
# run.
# n_ants = [4, 7, 8, 16, 32, 64, 80, 90, 128, 150, 256]
# n_ants = [7, 14, 21, 28]
# n_ants = []
# ants_min = 4
# for i in range(256):
#     n_ants.append(ants_min+i)

# n_ants = [80]
n_ants = [4, 8, 16, 32, 64, 79, 80, 84, 130, 192, 256, 5, 23, 61, 19]

"""n_samples_per_channel: List of number of samples per channel."""
# This is always set to 256 for the MeerKAT case and leads to packet sizes equal to 256 * 4 bytes per sample = 1 KiB.
# There is a chance that for the MeerKAT extension this could be configured to allow for packet sizes equal to 2 or 4
# KiB. All these cases need to be tested.
n_samples_per_channel = [256]

"""n_channels: List of number of total channels per array."""
# Number of FFT channels out of the F-Engine
# n_channels = [1024, 2048, 4096, 8192, 16384, 32768]
n_channels = [1024, 4096, 32768]

"""n_beams: List of number of beams."""
n_beams = [2]
xeng_id = [0]
samples_delay = [5]
phase = [np.pi / 2]
