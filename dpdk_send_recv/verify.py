#!/usr/bin/env python3

import argparse
import struct
import numpy as np

import pcap

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--packet-size", type=int, default=4096)
parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
parser.add_argument("--chunks", type=int, default=2)
args = parser.parse_args()

f = pcap.pcap(name=args.filename)
dtype = np.dtype(np.uint64)
chunk_id = 0
chunk_pos = 0
chunk_items = args.chunk_size // dtype.itemsize
for i, (ts, pkt) in enumerate(f):
    payload = pkt[42:]  # Strip off headers
    data = np.frombuffer(payload, dtype=np.uint64)
    expected = np.arange(0, args.packet_size // dtype.itemsize, dtype=dtype)
    expected += chunk_pos | (chunk_id << 32)
    expected[0] = i
    np.testing.assert_equal(expected, data)
    chunk_pos += len(expected)
    if chunk_pos >= chunk_items:
        chunk_pos -= chunk_items
        chunk_id += 1
        if chunk_id >= args.chunks:
            chunk_id = 0
