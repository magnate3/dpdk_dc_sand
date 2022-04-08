#!/usr/bin/env python3
################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Analyse dsim data."""

import argparse
import asyncio
import logging
from turtle import setundobuffer
from typing import List, Union

import aiokatcp
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
import spead2
import spead2.recv
import spead2.recv.asyncio
from katsdpservices import get_interface_address
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser, endpoint_parser
from numba import types
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data
from katgpucbf.ringbuffer import ChunkRingbuffer
from katgpucbf.fgpu import recv
from katgpucbf.monitor import NullMonitor
from katgpucbf import recv as base_recv
import katsdpsigproc.accel as accel
import allantools

CPLX = 2

def cw_scale():
    pass

def cw_linearity():
    pass


def allan_variance(samples):
    n = 3000
    taus = np.linspace(1,n,n)
    (t2, ad, ade, adn) = allantools.oadev(samples, rate=1, data_type="freq", taus=taus)
    plt.plot(t2,ad)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('WGN')
    plt.ylabel('Allan Deviation')
    plt.show()

@numba.njit
def unpackbits(packed_data: np.ndarray, unpacked_data_length) -> np.ndarray:
    """Unpack 8b data words to 10b data words.

    Parameters
    ----------
    packed_data
        A numpy ndarray of packed 8b data words.
    """
    unpacked_data = np.zeros((unpacked_data_length,), dtype=np.int16)
    data_sample = np.int16(0)
    pack_idx = 0
    unpack_idx = 0

    for pack_idx in range(0, len(packed_data), 5):
        tmp_40b_word = np.uint64(
            packed_data[pack_idx] << (8 * 4)
            | packed_data[pack_idx + 1] << (8 * 3)
            | packed_data[pack_idx + 2] << (8 * 2)
            | packed_data[pack_idx + 3] << 8
            | packed_data[pack_idx + 4]
        )

        for data_idx in range(4):
            data_sample = np.int16((tmp_40b_word & np.uint64(0xFFC0000000)) >> np.uint64(30))
            if data_sample > 511:
                data_sample = data_sample - 1024
            unpacked_data[unpack_idx + data_idx] = np.int16(data_sample)
            tmp_40b_word = tmp_40b_word << np.uint8(10)
        unpack_idx += 4
    return unpacked_data


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        type=get_interface_address,
        required=True,
        help="Name of network  to use for ingest.",
    )
    parser.add_argument(
        "--ibv",
        action="store_true",
        help="Use ibverbs",
    )
    # parser.add_argument(
    #     "--mc-address",
    #     type=endpoint_parser(5001),
    #     default="lab5.sdp.kat.ac.za:5001",  # Naturally this applies only to our lab...
    #     help="Master controller to query for details about the product. [%(default)s]",
    # )
    # parser.add_argument("product_name", type=str, help="Name of the subarray to get baselines from.")
    args = parser.parse_args()
    asyncio.run(async_main(args))

async def async_main(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)

    # dsim_host_data = "10.100.44.1"
    # dsim_port_data = 7148
    dsim_host_katcp =  "qgpu01.sdpdyn.kat.ac.za"
    dsim_port_katcp = 7147

    srcs = [
        [
            ("239.103.0.64", 7148),
            ("239.103.0.65", 7148),
            ("239.103.0.66", 7148),
            ("239.103.0.67", 7148),
            ("239.103.0.68", 7148),
            ("239.103.0.69", 7148),
            ("239.103.0.70", 7148),
            ("239.103.0.71", 7148),
        ],
        [
            ("239.103.0.72", 7148),
            ("239.103.0.73", 7148),
            ("239.103.0.74", 7148),
            ("239.103.0.75", 7148),
            ("239.103.0.76", 7148),
            ("239.103.0.77", 7148),
            ("239.103.0.78", 7148),
            ("239.103.0.79", 7148),
        ]
    ]

    dsim_client = await aiokatcp.Client.connect(dsim_host_katcp, dsim_port_katcp)
    logger.info("Successfully connected to dsim.")
    
    freq = 10e6
    logger.info("Setting dsim cw freq to %f", freq)
    [reply, _informs] = await dsim_client.request("signals", f"common=cw(0.0,{freq})+wgn(0.05);common;common;")
    expected_timestamp = int(reply[0])

    src_packet_samples = 4096
    chunk_samples = 2**16 #2**24
    mask_timestamp = False
    SAMPLE_BITS = 10
    N_POLS = 2
    src_affinity = [-1] * N_POLS
    src_comp_vector = [-1] * N_POLS
    src_buffer = 32 * 1024 * 1024
    layout = recv.Layout(SAMPLE_BITS, src_packet_samples, chunk_samples, mask_timestamp)

    BYTE_BITS = 8
    ringbuffer_capacity = 2
    ring = ChunkRingbuffer(ringbuffer_capacity, name="recv_ringbuffer", task_name="run_receive", monitor=NullMonitor())
    
    streams = recv.make_streams(layout, ring, src_affinity)

    # Option 1:
    #----------
    # for pol, stream in enumerate(streams):
    #     base_recv.add_reader(
    #         stream,
    #         src=srcs[pol],
    #         interface=args.interface,
    #         ibv=args.ibv,
    #         comp_vector=src_comp_vector[pol],
    #         buffer=src_buffer,
    #     )

    # Option 2:
    #----------
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    src_chunks_per_stream = 4   
    chunk_bytes = chunk_samples * SAMPLE_BITS // BYTE_BITS
    for pol, stream in enumerate(streams):
        for _ in range(src_chunks_per_stream):
            buf = accel.HostArray((chunk_bytes,), np.uint8, context=ctx)
            chunk = recv.Chunk(data=buf)
            chunk.present = np.zeros(chunk_samples // src_packet_samples, np.uint8)
            stream.add_free_chunk(chunk)

    for pol, stream in enumerate(streams):
        base_recv.add_reader(
            stream,
            src=srcs[pol],
            interface=args.interface,
            ibv=args.ibv,
            comp_vector=src_comp_vector[pol],
            buffer=src_buffer,
        )

    print('Post add reader')
    frequency_to_check = [100e6, 200e6]
    for freq in frequency_to_check:
        # Dsim set freq
        expected_timestamp = int(reply[0])
        print(f'Expected time {expected_timestamp}')

        async for chunks in recv.chunk_sets(streams, layout):
            print('Rx')
            if not np.all(chunks[0].present) and np.all(chunks[1].present):
                logger.debug("Incomplete chunk %d", chunks[0].chunk_id)
                stream.add_free_chunk(chunks)
            elif chunks[0].timestamp <= expected_timestamp:
                # logger.info("Skipping chunk with timestamp %d", recvd_timestamp)
                streams.add_free_chunk(chunks)
            else:
                pol0_data = chunks[0].data  #This is 8b data
                
                # unpack data to 10b
                print(f'length is {len(pol0_data)}')
                recon_data = unpackbits(pol0_data, chunk_samples)
                print(f'length is {len(recon_data)}')
                
                # Allan variance
                allan_variance(pol0_data)

                # do some stuff 

                # plt.figure(1)
                # plt.plot(recon_data)
                # plt.show()

                for pol in range(len(chunks)):
                    streams[pol].add_free_chunk(chunks[pol])

if __name__ == "__main__":
    main()