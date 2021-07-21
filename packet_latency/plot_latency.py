#!/usr/bin/env python3

"""See README.md."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--adc-sample-rate", type=float, default=1712e6)
parser.add_argument("--decimate", type=int, default=32)
parser.add_argument("input")
args = parser.parse_args()

data = np.load(args.input, allow_pickle=False)
adc_timestamps = data["adc_timestamps"]
adc_timestamps -= adc_timestamps[0]
pkt_timestamps = data["pkt_timestamps"]
pkt_timestamps -= pkt_timestamps[0]

delays = pkt_timestamps / 1e9 - adc_timestamps / args.adc_sample_rate

plt.plot(adc_timestamps[:: args.decimate] / args.adc_sample_rate, delays[:: args.decimate], ".-")
plt.xlabel("ADC time [s]")
plt.ylabel("Pkt time - ADC time + C [s]")
plt.show()
