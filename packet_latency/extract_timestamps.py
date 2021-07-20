#!/usr/bin/env python3

"""See README.md."""

import argparse
import struct

import numpy as np
import pcap

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

f = pcap.pcap(name=args.input, timestamp_in_ns=True)
pkt_timestamps = []
adc_timestamps = []
for pkt_ts, pkt in f:
    payload = pkt[42:]
    items = struct.unpack(">H", payload[6:8])[0]
    adc_ts = -1
    for i in range(items):
        item_ptr = struct.unpack(">Q", payload[8 * (i + 1) : 8 * (i + 2)])[0]
        item_id = (item_ptr >> 48) & 0x1FFF
        item_data = item_ptr & ((1 << 48) - 1)
        if item_id == 0x1600:  # timestamp
            adc_ts = item_data
            break
    if adc_ts == -1:
        continue
    adc_timestamps.append(adc_ts)
    pkt_timestamps.append(pkt_ts)

np.savez(args.output, pkt_timestamps=np.array(pkt_timestamps), adc_timestamps=np.array(adc_timestamps))
