/* Example application using DPDK to send arbitrary data on a network.
 * Written to learn about DPDK - not meant to be an example of good code.
 */

#include <iostream>
#include <cstring>
#include <cstdint>

#include <rte_byteorder.h>
#include <rte_debug.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_flow.h>
#include <rte_ip.h>

#include "dpdk_common.h"

static void append_to_mbuf(rte_mbuf *mbuf, const void *data, std::size_t size)
{
    void *ptr = rte_pktmbuf_append(mbuf, size);
    if (!ptr)
        rte_panic("rte_pktmbuf_append failed\n");
    std::memcpy(ptr, data, size);
}

struct prepare_mbuf_context
{
    const device_info *info;
    std::uint16_t payload_size;
};

/* Fill in L2-L4 headers. The signature is to conform to the callback for
 * rte_mempool_obj_iter.
 */
static void prepare_mbuf(rte_mempool *mp, void *data, void *obj, unsigned obj_idx)
{
    rte_mbuf *mbuf = (rte_mbuf *) obj;
    const prepare_mbuf_context &ctx = *(prepare_mbuf_context *) data;

    rte_ether_hdr ether_hdr = {
        .dst_addr = MULTICAST_MAC,
        .src_addr = ctx.info->mac,
        .ether_type = RTE_BE16(RTE_ETHER_TYPE_IPV4)
    };

    rte_ipv4_hdr ipv4_hdr = {
        .version_ihl = 0x45,  // version 4, 20-byte header
        .total_length = rte_cpu_to_be_16(ctx.payload_size + sizeof(rte_udp_hdr) + sizeof(rte_ipv4_hdr)),
        .fragment_offset = RTE_BE16(0x4000),    // Don't-fragment
        .time_to_live = 4,
        .next_proto_id = IPPROTO_UDP,
        .src_addr = rte_cpu_to_be_32(ctx.info->ipv4_addr),
        .dst_addr = MULTICAST_GROUP
    };
    ipv4_hdr.hdr_checksum = rte_ipv4_cksum(&ipv4_hdr);

    // TODO: get a valid src port number from the kernel?
    rte_udp_hdr udp_hdr = {
        .src_port = rte_cpu_to_be_16(1234),
        .dst_port = MULTICAST_PORT,
        .dgram_len = rte_cpu_to_be_16(ctx.payload_size + sizeof(rte_udp_hdr)),
        .dgram_cksum = 0
    };

    rte_pktmbuf_reset(mbuf);
    append_to_mbuf(mbuf, &ether_hdr, sizeof(ether_hdr));
    append_to_mbuf(mbuf, &ipv4_hdr, sizeof(ipv4_hdr));
    append_to_mbuf(mbuf, &udp_hdr, sizeof(udp_hdr));
}

int main(int argc, char **argv)
{
    int ret;

    ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_panic("Cannot init EAL\n");

    device_info info = choose_device();
    std::cout << "Found device with driver name " << info.dev_info.driver_name
        << ", interface " << (info.ifname.empty() ? "none" : info.ifname) << "\n";

    /* Ignore any rx traffic which doesn't match a flow rule
     * (i.e., all of it, since none get set up).
     */
    ret = rte_flow_isolate(info.port_id, 1, NULL);
    if (ret != 0)
        rte_panic("rte_flow_isolate failed\n");

    rte_eth_conf eth_conf = {};
    ret = rte_eth_dev_configure(info.port_id, 1, 1, &eth_conf);
    if (ret != 0)
        rte_panic("rte_eth_dev_configure failed\n");

    uint16_t nb_rx_desc = 1;
    uint16_t nb_tx_desc = 128;
    rte_eth_dev_adjust_nb_rx_tx_desc(info.port_id, &nb_rx_desc, &nb_tx_desc);

    // * 2 - 1 to hopefully get one less than a power of 2, which is apparently optimal
    int socket_id = rte_eth_dev_socket_id(info.port_id);
    rte_mempool *send_mb_pool = rte_pktmbuf_pool_create("send", nb_tx_desc * 2 - 1, 0, 0, 16384, socket_id);
    if (!send_mb_pool)
        rte_panic("rte_pktmbuf_pool_create failed\n");
    std::uint64_t payload[4] = {};
    prepare_mbuf_context ctx = {&info, sizeof(payload)};
    rte_mempool_obj_iter(send_mb_pool, prepare_mbuf, &ctx);

    // TODO: does IP checksum offload need to be enabled?
    rte_eth_txconf tx_conf = {};  // TODO use dev_info.default_txconf?
    ret = rte_eth_tx_queue_setup(info.port_id, 0, nb_tx_desc, socket_id, &tx_conf);
    if (ret != 0)
        rte_panic("rte_eth_tx_queue_setup failed\n");

    /* Don't actually want any RX, but can't set 0 queues. We also
     * have to give it a packet pool. The data_room_size is big to
     * ensure it exceeds the MTU.
     */
    rte_mempool *recv_mb_pool = rte_pktmbuf_pool_create("recv", 127, 0, 0, 16384, SOCKET_ID_ANY);
    if (!recv_mb_pool)
        rte_panic("rte_pktmbuf_pool_create failed\n");
    ret = rte_eth_rx_queue_setup(info.port_id, 0, nb_rx_desc, SOCKET_ID_ANY, NULL, recv_mb_pool);
    if (ret != 0)
        rte_panic("rte_eth_rx_queue_setup failed\n");

    ret = rte_eth_dev_start(info.port_id);
    if (ret != 0)
        rte_panic("rte_eth_dev_start failed\n");

    for (std::uint64_t cnt = 0; ; cnt++)
    {
        payload[0] = cnt;

        rte_mbuf *mbuf = rte_pktmbuf_alloc(send_mb_pool);
        rte_pktmbuf_reset(mbuf);
        /* Extend data pointer to encompass the pre-written headers */
        rte_pktmbuf_append(mbuf, sizeof(rte_ether_hdr) + sizeof(rte_ipv4_hdr) + sizeof(rte_udp_hdr));
        append_to_mbuf(mbuf, &payload, sizeof(payload));

        /* Send 1 packet. If the queue is full, try again.
         */
        do
        {
            ret = rte_eth_tx_burst(info.port_id, 0, &mbuf, 1);
        } while (ret == 0);
    }

    rte_eal_cleanup();
    return 0;
}
