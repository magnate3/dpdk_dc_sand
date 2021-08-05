# Measuring packet latency for SPEAD data

This directory contains utilities for extracting timestamps from the SPEAD
items in pcap files and comparing them to packet arrival timestamps.

## Process

1. Install dependencies: pypcap, numpy, matplotlib.

2. Capture some data to a pcap file.
   [mcdump](https://spead2.readthedocs.io/en/latest/tools.html#mcdump) is a
   useful tool for this.

3. Extract the timestamps from the pcap metadata and the SPEAD packets:

       ./extract_timestamps.py raw.pcap timestamps.npz

   The timestamps.npz file contains two numpy arrays, one with packet
   timestamps (in nanoseconds) and one with the value of the timestamp item
   (ID 0x1600) from the packets. Packets without this item are skipped.

4. Plot the relative latencies (shows an interactive Matplotlib window):

       ./plot_latency.py --adc-sample-rate 1712e6 timestamps.npz

   At present the latencies are relative to that of the first packet, so it
   is useful for examining jitter rather than absolute latencies. If absolute
   latencies are required, the script will need to be extended to take a sync
   epoch for the ADC timestamps.

   There is a `--decimate` option that controls what fraction of the packets to
   plot, which allows large numbers of packets to be shown without overloading
   Matplotlib.
