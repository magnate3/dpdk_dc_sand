/* Example application using DPDK to receive arbitrary multicast data from a
 * network.
 * Written to learn about DPDK - not meant to be an example of good code.
 */

#include <iostream>

#include <rte_byteorder.h>
#include <rte_debug.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_flow.h>

#include "dpdk_common.h"

static rte_flow *create_flow(std::uint16_t port_id, rte_flow_error *flow_error)
{
    const rte_flow_attr flow_attr = {.ingress = 1};
    const rte_flow_item_eth eth_spec = {
        .hdr = {
            .dst_addr = {{0x01, 0x00, 0x5E, 0x66, 0x11, 0x12}},
            .ether_type = RTE_BE16(RTE_ETHER_TYPE_IPV4)
        }
    };
    const rte_flow_item_eth eth_mask = {
        .hdr = {
            .dst_addr = {{0xff, 0xff, 0xff, 0xff, 0xff, 0xff}},
            .ether_type = RTE_BE16(0xffff)
        },
    };
    const rte_flow_item_ipv4 ipv4_spec = {
        .hdr = {
            .dst_addr = RTE_BE32(RTE_IPV4(239, 102, 17, 18))
        }
    };
    const rte_flow_item_ipv4 ipv4_mask = {
        .hdr = {
            .dst_addr = RTE_BE32(0xffffffff)
        }
    };
    const rte_flow_item_udp udp_spec = {
        .hdr = {
            .dst_port = RTE_BE16(8888)
        }
    };
    const rte_flow_item_udp udp_mask = {
        .hdr = {
            .dst_port = RTE_BE16(0xffff)
        }
    };
    const rte_flow_item pattern[] = {
        {
            .type = RTE_FLOW_ITEM_TYPE_ETH,
            .spec = &eth_spec,
            .mask = &eth_mask
        },
        {
            .type = RTE_FLOW_ITEM_TYPE_IPV4,
            .spec = &ipv4_spec,
            .mask = &ipv4_mask
        },
        {
            .type = RTE_FLOW_ITEM_TYPE_UDP,
            .spec = &udp_spec,
            .mask = &udp_mask
        },
        {
            .type = RTE_FLOW_ITEM_TYPE_END
        }
    };
    const rte_flow_action_queue queue_action = {
        .index = 0   // rx queue index
    };
    const rte_flow_action action[] = {
        {
            .type = RTE_FLOW_ACTION_TYPE_QUEUE,
            .conf = &queue_action
        },
        {
            .type = RTE_FLOW_ACTION_TYPE_END
        }
    };
    return rte_flow_create(port_id, &flow_attr, pattern, action, flow_error);
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

    ret = rte_flow_isolate(info.port_id, 1, NULL);
    if (ret != 0)
        rte_panic("rte_flow_isolate failed\n");

    rte_eth_conf eth_conf = {};
    ret = rte_eth_dev_configure(info.port_id, 1, 1, &eth_conf);
    if (ret != 0)
        rte_panic("rte_eth_dev_configure failed\n");

    uint16_t nb_rx_desc = 128;
    uint16_t nb_tx_desc = 1;
    rte_eth_dev_adjust_nb_rx_tx_desc(info.port_id, &nb_rx_desc, &nb_tx_desc);

    /* * 2 - 1 to hopefully get one less than a power of 2, which is apparently optimal.
     * TODO: figure out appropriate size.
     * TODO: cache?
     */
    int socket_id = rte_eth_dev_socket_id(info.port_id);
    rte_mempool *recv_mb_pool = rte_pktmbuf_pool_create("recv", nb_rx_desc * 2 - 1, 0, 0, 16384, socket_id);
    if (!recv_mb_pool)
        rte_panic("rte_pktmbuf_pool_create failed\n");
    ret = rte_eth_rx_queue_setup(info.port_id, 0, nb_rx_desc, socket_id, NULL, recv_mb_pool);
    if (ret != 0)
        rte_panic("rte_eth_rx_queue_setup failed\n");

    ret = rte_eth_dev_start(info.port_id);
    if (ret != 0)
        rte_panic("rte_eth_dev_start failed\n");

    rte_flow_error flow_error;
    rte_flow *flow = create_flow(info.port_id, &flow_error);
    if (!flow)
    {
        std::cerr << "cause: " << flow_error.cause << '\n'
            << "message: " << flow_error.message << '\n';
        rte_panic("rte_flow_create failed\n");
    }

    while (true)
    {
        constexpr int max_pkts = 32;
        rte_mbuf *rx_pkts[max_pkts];
        int pkts = rte_eth_rx_burst(info.port_id, 0, rx_pkts, max_pkts);
        if (pkts)
        {
            std::cout << "Received burst of " << pkts << " packets\n";
            for (int i = 0; i < pkts; i++)
            {
                char buf[128];
                rte_get_rx_ol_flag_list(rx_pkts[i]->ol_flags, buf, sizeof(buf));
                std::cout << "Packet with length " << rx_pkts[i]->pkt_len << ", flags " << buf << '\n';
            }
            rte_pktmbuf_free_bulk(rx_pkts, pkts);
        }
    }

    rte_flow_flush(info.port_id, NULL);
    return 0;
}
