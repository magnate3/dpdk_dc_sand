"""Module containing correlator configuration options that should be tested where appropriate in all unit tests."""
import numpy as np

# """batches: Number of batches for processing."""
batches = [3]

"""array_size: List of array sizes."""
# These are the estimated subarray sizes that will be run. The 130, 192 and 256 values are estimates for SKA and are
# not final. Additionally values 5,23,61 and 19 are just there to test that various non-power-of-two array sizes will
# run.
array_size = [4, 8, 16, 32, 64, 128, 256]
# array_size = [128]
# array_size = [130, 192, 256, 5, 23, 61, 19]
# array_size = [4, 8, 16, 32, 64, 84, 130, 192, 256, 5, 23, 61, 19]

"""num_samples_per_channel: List of number of samples per channel."""
# This is always set to 256 for the MeerKAT case and leads to packet sizes equal to 256 * 4 bytes per sample = 1 KiB.
# There is a chance that for the MeerKAT extension this could be configured to allow for packet sizes equal to 2 or 4
# KiB. All these cases need to be tested.
# num_samples_per_channel = [256, 512, 1024]
num_samples_per_channel = [256]

"""num_channels: List of number of total channels per array."""
# Number of FFT channels out of the F-Engine
# num_channels = [1024, 4096, 32768]
num_channels = [1024]
# num_channels = [1024, 4096]


"""num_beams: List of number of beams."""
num_beams = [8]
xeng_id = [0]
samples_delay = [5]
phase = [np.pi/2]