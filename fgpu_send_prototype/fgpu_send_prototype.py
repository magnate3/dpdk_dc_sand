#!/usr/bin/env python3

"""
Proof-of-concept for transmit in katgpucbf.fgpu.

To get good performance, the heaps are all created in advance and carefully set
up to reference mutable memory. To send new data, the numpy arrays are updated
without needing to create new heaps.
"""

import argparse
import asyncio
import time

import numpy as np
import spead2.send.asyncio
from katsdptelstate.endpoint import endpoint_parser

FLAVOUR = spead2.Flavour(4, 64, 48, 0)
TIMESTAMP_ID = 0x1600
FREQUENCY_ID = 0x4103
FENG_RAW_ID = 0x4300


def make_immediate(id: int, value: np.ndarray) -> spead2.Item:
    """Synthesize an immediate item that references (not copies) some memory.

    This is suitable for adding directly to a heap, but not for descriptors.

    Parameters
    ----------
    id
        The SPEAD identifier for the item
    value
        A single-element array with dtype ``>u8``
    """
    assert value.dtype == np.dtype(">u8")
    assert value.shape == (1,)
    # Access the raw bytes, so that we can select just the right number of LSBs.
    n_bytes = FLAVOUR.heap_address_bits // 8
    imm_value = value.view(np.uint8)[-n_bytes:]
    return spead2.Item(id, "dummy_item", "", (n_bytes,), dtype=imm_value.dtype, value=imm_value)


class Frame:
    """Hold all data and heaps for one timestamp."""

    def __init__(self, n_heaps, heap_size):
        self.timestamp = np.ones(1, dtype=">u8")
        self.payload = np.ones((n_heaps, heap_size), dtype=np.uint8)
        self.transmit_future = asyncio.get_event_loop().create_future()
        self.transmit_future.set_result(None)
        self.heaps = []
        for i in range(n_heaps):
            frequency = np.array([i * 4], dtype=">u8")
            heap = spead2.send.Heap(FLAVOUR)
            payload = self.payload[i]
            heap.add_item(make_immediate(TIMESTAMP_ID, self.timestamp))
            heap.add_item(make_immediate(FREQUENCY_ID, frequency))
            heap.add_item(spead2.Item(FENG_RAW_ID, "feng_raw", "", payload.shape, dtype=payload.dtype, value=payload))
            self.heaps.append(spead2.send.HeapReference(heap))


async def main():  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", required=True)
    parser.add_argument("--parallel", type=int, default=256)
    parser.add_argument("--heap-size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--rate", type=float, help="Rate in Gb/s")
    parser.add_argument("--ibv", action="store_true")
    parser.add_argument("destination", type=endpoint_parser(7148))
    args = parser.parse_args()

    frames = [Frame(args.parallel, args.heap_size) for _ in range(args.depth)]
    config = spead2.send.StreamConfig(max_packet_size=8192 + 128, max_heaps=args.depth * args.parallel)
    if args.rate:
        config.rate = args.rate * 0.125e9
    if args.ibv:
        ibv_config = spead2.send.UdpIbvConfig(
            endpoints=[tuple(args.destination)],
            interface_address=args.bind,
            ttl=4,
            memory_regions=[frame.payload for frame in frames],
        )
        stream = spead2.send.asyncio.UdpIbvStream(spead2.ThreadPool(), config, ibv_config)
    else:
        stream = spead2.send.asyncio.UdpStream(
            spead2.ThreadPool(), [tuple(args.destination)], config, interface_address=args.bind
        )

    # Create an item group purely to get descriptors
    ig = spead2.send.ItemGroup()
    imm_format = [("u", FLAVOUR.heap_address_bits)]
    ig.add_item(TIMESTAMP_ID, "timestamp", "timestamp description", (), format=imm_format)
    ig.add_item(FREQUENCY_ID, "frequency", "frequency description", (), format=imm_format)
    ig.add_item(FENG_RAW_ID, "feng_raw", "feng_raw description", (args.heap_size,), dtype=np.int8)
    await stream.async_send_heap(ig.get_heap(descriptors="all", data="none"))

    start = time.monotonic()
    for i in range(args.steps):
        frame = frames[i % args.depth]
        await frame.transmit_future
        frame.timestamp[:] = i
        frame.payload[:] = i
        frame.transmit_future = asyncio.ensure_future(
            stream.async_send_heaps(frame.heaps, spead2.send.GroupMode.ROUND_ROBIN)
        )
    await stream.async_flush()
    stop = time.monotonic()
    elapsed = stop - start
    rate = (args.steps * args.parallel * args.heap_size * 8) / elapsed
    print(f"{rate / 1e9} Gb/s")


if __name__ == "__main__":
    asyncio.run(main())
