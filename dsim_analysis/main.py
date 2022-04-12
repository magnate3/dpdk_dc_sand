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
import config
from config import CPLX, BYTE_BITS, SAMPLE_BITS, N_POLS
import wgn, allan_var
import katsdpsigproc.accel as accel

@numba.njit
async def unpackbits(packed_data: np.ndarray, unpacked_data_length) -> np.ndarray:
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
    args = parser.parse_args()
    asyncio.run(async_main(args))

async def async_main(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)

    dsim_client = await aiokatcp.Client.connect(config.dsim_host_katcp, config.dsim_port_katcp)
    logger.info("Successfully connected to dsim.")
    
    freq = 10e6
    logger.info("Setting dsim cw freq to %f", freq)
    [reply, _informs] = await dsim_client.request("signals", f"common=cw(0.0,{freq})+wgn(0.05);common;common;")
    expected_timestamp = int(reply[0])

    src_packet_samples = 4096
    chunk_samples = 2**18 #2**24
    mask_timestamp = False

    src_affinity = [-1] * N_POLS
    src_comp_vector = [-1] * N_POLS
    src_buffer = 32 * 1024 * 1024
    layout = recv.Layout(SAMPLE_BITS, src_packet_samples, chunk_samples, mask_timestamp)

    ringbuffer_capacity = 2
    ring = ChunkRingbuffer(ringbuffer_capacity, name="recv_ringbuffer", task_name="run_receive", monitor=NullMonitor())
    
    streams = recv.make_streams(layout, ring, src_affinity)

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
            src=config.srcs[pol],
            interface=args.interface,
            ibv=args.ibv,
            comp_vector=src_comp_vector[pol],
            buffer=src_buffer,
        )

    # Start tests
    # -----------
    # wgn_tests = wgn.wgn_analysis()

    
    for freq in config.frequencies_to_check:
        print(f'*** Freq is {freq} ***')
        logger.info("Setting dsim cw freq to %f", freq)
        pass
    
    wgn_test_results = []
    for wgn_scale in config.noise_scales:
        cw_scale = 0.0
        [reply, _informs] = await dsim_client.request("signals", f"common=cw({cw_scale},{freq})+wgn({wgn_scale});common;common;")
        expected_timestamp = int(reply[0])

        count = 0
        recon_data = []
        async for chunks in recv.chunk_sets(streams, layout):
            # print('Rx')
            if not np.all(chunks[0].present) and np.all(chunks[1].present):
                logger.debug("Incomplete chunk %d", chunks[0].chunk_id)
                for pol in range(len(chunks)):
                    streams[pol].add_free_chunk(chunks[pol])
            elif chunks[0].timestamp <= expected_timestamp:
                # logger.info("Skipping chunk with timestamp %d", recvd_timestamp)
                for pol in range(len(chunks)):
                    streams[pol].add_free_chunk(chunks[pol])
            else:
                # pol0_data = chunks[0].data  #This is 8b data
                # pol1_data = chunks[1].data  #This is 8b data

                # if (count % 100)==False:
                #     print(count)
                # count += 1

                # unpack data to 10b
                # print(f'length is {len(pol0_data)}')
                for pol in range(len(chunks)):
                    unpacked_data = unpackbits(chunks[pol].data, chunk_samples)
                    unpacked_data = unpacked_data/(np.abs(np.max(unpacked_data)))
                    recon_data.append(unpacked_data)
                    # recon_data.append(unpackbits(chunks[pol].data, chunk_samples))
                  
                # WGN tests
                mean, std_dev, var, hist, allan_var = await wgn.wgn_analysis.run(recon_data)
                wgn_test_results.append((wgn_scale, mean, std_dev, var, hist, allan_var))
                
                for pol in range(len(chunks)):
                    streams[pol].add_free_chunk(chunks[pol])
                break
    a = 1

if __name__ == "__main__":
    main()