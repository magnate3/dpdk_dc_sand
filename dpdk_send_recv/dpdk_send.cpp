/* Example application using DPDK to send arbitrary data on a network.
 * Written to learn about DPDK - not meant to be an example of good code.
 */

#include <iostream>
#include <net/if.h>

#include <rte_eal.h>
#include <rte_debug.h>
#include <rte_ethdev.h>

int main(int argc, char **argv)
{
    int ret;
    uint16_t port_id;
    bool found = false;

    ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_panic("Cannot init EAL\n");

    RTE_ETH_FOREACH_DEV(port_id)
    {
        rte_eth_dev_info dev_info;
        ret = rte_eth_dev_info_get(port_id, &dev_info);
        if (ret != 0)
            rte_panic("rte_eth_dev_info_get failed\n");
        char ifname_storage[IF_NAMESIZE];
        const char *ifname;
        if (dev_info.if_index > 0)
        {
            ifname = if_indextoname(dev_info.if_index, ifname_storage);
            if (ifname == NULL)
                ifname = "none";
        }
        std::cout << "Found device with driver name " << dev_info.driver_name << ", interface " << ifname << "\n";
        found = true;
        break;
    }
    if (!found)
        rte_panic("no devices found\n");

    rte_eth_conf eth_conf = {};
    ret = rte_eth_dev_configure(port_id, 1, 1, &eth_conf);
    if (ret != 0)
        rte_panic("rte_eth_dev_configure failed\n");

    uint16_t nb_rx_desc = 1;
    uint16_t nb_tx_desc = 128;
    rte_eth_dev_adjust_nb_rx_tx_desc(port_id, &nb_rx_desc, &nb_tx_desc);

    // TODO: does IP checksum offload need to be enabled?
    rte_eth_txconf tx_conf = {};
    ret = rte_eth_tx_queue_setup(port_id, 0, nb_tx_desc, rte_socket_id(), &tx_conf);
    if (ret != 0)
        rte_panic("rte_eth_tx_queue_setup failed\n");

    /* Don't actually want any RX, but can't set 0 queues. We also
     * have to give it a packet pool. The data_room_size is big to
     * ensure it exceeds the MTU.
     */
    rte_mempool *recv_mb_pool = rte_pktmbuf_pool_create("recv", 127, 0, 0, 16384, SOCKET_ID_ANY);
    if (!recv_mb_pool)
        rte_panic("rte_pktmbuf_pool_create failed");
    ret = rte_eth_rx_queue_setup(port_id, 0, nb_rx_desc, SOCKET_ID_ANY, NULL, recv_mb_pool);
    if (ret != 0)
        rte_panic("rte_eth_rx_queue_setup failed\n");

    ret = rte_eth_dev_start(port_id);
    if (ret != 0)
        rte_panic("rte_eth_dev_start failed\n");

    rte_eal_cleanup();
    return 0;
}
