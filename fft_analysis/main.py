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
from dataclasses import dataclass
import logging
import aiokatcp
import matplotlib.pyplot as plt
import numba
import numpy as np
import spead2.recv
import spead2.recv.asyncio
from katsdpservices import get_interface_address
from katgpucbf.ringbuffer import ChunkRingbuffer
from katgpucbf.fgpu import recv
from katgpucbf.monitor import NullMonitor
from katgpucbf import recv as base_recv
import config
from config import  BYTE_BITS, SAMPLE_BITS, N_POLS
import katsdpsigproc.accel as accel
import fft_compute
import time
import katsdpsigproc.accel
from katsdpsigproc.accel import visualize_operation
import fft_standalone_fp16_fp32
import cupy as cp

def _pack_real_to_complex(data_in):
    shape = np.shape(data_in)
    data_out = np.zeros((shape[0], shape[1], 2*shape[2]), dtype=np.float16)
    r = 0
    for n in range(shape[2]):
        data_out[0][0][r] = data_in[0][0][n]
        r +=2
    return data_out

def fft_gpu_fp16(data):
    # shape = np.shape(data)
    # a_real = cp.zeros((shape[0], shape[1], 2*shape[2]), dtype=np.float16)

    # Pack real data into complex format for FP16 FFT
    _pack_real_to_complex(data)
    a_real = cp.array(data)
    return fft_standalone_fp16_fp32._fft_fp16_gpu(a_real)

def fft_cpu(data):
    # shape = np.shape(data)
    # a_real = cp.zeros((shape[0], shape[1], 2*shape[2]), dtype=np.float16)

    # Pack real data into complex format for FP16 FFT
    a_real = _pack_real_to_complex(data)
    return fft_standalone_fp16_fp32._fft_cpu(a_real)

def fft_gpu_fp32(data):
    return fft_standalone_fp16_fp32._fft_gpu_fp32(data)

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
    start_time = time.time()
    logger = logging.getLogger(__name__)

    # Create compute context
    ctx = accel.create_some_context(
        device_filter=lambda x: x.is_cuda, interactive=False
    )
    queue = ctx.create_command_queue()

    # Import FFT template
    template = fft_compute.ComputeTemplate(ctx)
    op = template.instantiate(queue, 4194304, 1, 256, 32768)
    op.ensure_all_bound()

    # Visualise the operation
    katsdpsigproc.accel.visualize_operation(op,'test_op_vis')

    buf_fft_in = []
    host_fft_in = []
    for pol in range(N_POLS):
        buf_fft_in.append(op.fft[pol].buffer("src"))
        host_fft_in.append(buf_fft_in[pol].empty_like())
    
        buf_fft_out_device = op.fft[pol].buffer("dest")
        host_fft_out = buf_fft_out_device.empty_like()
    

    # Create connection to running DSim
    dsim_client = await aiokatcp.Client.connect(config.dsim_host_katcp, config.dsim_port_katcp)
    logger.info("Successfully connected to dsim.")
    
    # Start tests
    # -----------
    current_test = ''

    for value_set in config.value_sets:
        test = value_set[0]
        cw_scale = value_set[1]
        freq = value_set[2]
        wgn_scale = value_set[3]
        chunk_samples = value_set[4]

        print(f'Test is: {test} - CW Scale: {cw_scale}  Freq: {freq}  WGN_Scale:{wgn_scale}')

        if current_test != test:
            current_test = test
            print(f'Test is:{current_test} and Chunk size is: {chunk_samples}')

            src_packet_samples = 4096
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

        # Request DSim data
        # -----------------
        reply = []
        [reply, _informs] = await dsim_client.request("signals", f"common=wgn({wgn_scale});cw({cw_scale},{freq})+wgn({wgn_scale});cw({cw_scale},{freq})+wgn({wgn_scale});")
	    
        if reply == []:
            expected_timestamp = 0
        else:
            expected_timestamp = int(reply[0])

        recon_data = []
        async for chunks in recv.chunk_sets(streams, layout):

            if not np.all(chunks[0].present) and np.all(chunks[1].present):
            # if not (np.all(chunks[0].present) and np.all(chunks[1].present)):
                logger.debug("Incomplete chunk %d", chunks[0].chunk_id)
                for pol in range(len(chunks)):
                    streams[pol].add_free_chunk(chunks[pol])
            elif chunks[0].timestamp <= expected_timestamp:
                for pol in range(len(chunks)):
                    streams[pol].add_free_chunk(chunks[pol])
            else:
                # unpack data to 10b
                for pol in range(len(chunks)):
                    unpacked_data = unpackbits(chunks[pol].data, chunk_samples)
                    unpacked_data = unpacked_data/511
                    recon_data.append(unpacked_data)
                  
                # FFT
                for pol in range(N_POLS):
                    host_fft_in[pol][:] = recon_data[pol][:65536].reshape(1,65536)
                    buf_fft_in[pol].set(queue, host_fft_in[pol])

                # Run some FFT methods

                cpu_fft_out = fft_cpu(recon_data[0][:4096].reshape(1,1,4096))

                fft_gpu_fp16_out = fft_gpu_fp16(recon_data[0][:4096].reshape(1,1,4096))

                fft_gpu_fp32_out = fft_gpu_fp32(recon_data[0][:4096].reshape(1,1,4096))

                # or, run the op sequence....

                # Run the operational sequence          
                # op()

                # Grab the channelised data
                # buf_fft_out_device.get(queue, host_fft_out)


                # Plot incoming data
                # plt.figure(1)
                # plt.plot(recon_data[0])
                # plt.plot(recon_data[1])
                # plt.title('Pol0 and Pol1')
                # plt.show()

                # plt.figure(1)
                # plt.plot(10*np.log10(np.power(np.abs(host_fft_out[0]),2)))
                # plt.show()

                plt.figure(2)
                plt.plot(10*np.log10(np.power(np.abs(fft_gpu_fp16_out),2)))
                plt.plot(10*np.log10(np.power(np.abs(cpu_fft_out),2)))
                plt.show()


                for pol in range(len(chunks)):
                    streams[pol].add_free_chunk(chunks[pol])
                break
    
    print(f'Total execution time:{time.time() - start_time}')

    # Report Results
    a = 1

if __name__ == "__main__":
    main()