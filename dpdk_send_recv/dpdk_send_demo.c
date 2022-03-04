/* Simple application to demonstrate failure to transmit external memory.
 *
 * It attempts to send an IPv4 multicast packet with the headers in a
 * regular mbuf and the payload in a chained external mbuf.
 */

#define USE_EXTERNAL 1

#include <string.h>
#include <stdint.h>

#include <unistd.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <sys/mman.h>

#include <rte_byteorder.h>
#include <rte_debug.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_flow.h>
#include <rte_ip.h>
#include <rte_pause.h>

static const int PAYLOAD_SIZE = 1024;
static const struct rte_ether_addr MULTICAST_MAC = {{0x01, 0x00, 0x5E, 0x66, 0x11, 0x12}};
static const rte_be32_t MULTICAST_GROUP = RTE_BE32(RTE_IPV4(239, 102, 17, 18));
static const rte_be16_t MULTICAST_PORT = RTE_BE16(8888);
static const rte_be16_t SRC_PORT = RTE_BE16(1234);

uint16_t choose_device(struct rte_ether_addr *mac, rte_be32_t *ipv4_addr)
{
    bool found = false;
    int ret;
    uint16_t port_id = 0;
    struct rte_eth_dev_info dev_info;
    char ifname_storage[IF_NAMESIZE];
    const char *ifname = NULL;
    RTE_ETH_FOREACH_DEV(port_id)
    {
        ret = rte_eth_dev_info_get(port_id, &dev_info);
        if (ret != 0)
            rte_panic("rte_eth_dev_info_get failed\n");
        // If it corresponds to a kernel interface, we can get the name
        if (dev_info.if_index > 0)
        {
            ifname = if_indextoname(dev_info.if_index, ifname_storage);
        }
        port_id = port_id;
        found = true;
        break;
    }
    if (!found)
        rte_panic("no devices found\n");

    // Get the MAC address
    ret = rte_eth_macaddr_get(port_id, mac);
    if (ret != 0)
        rte_panic("rte_eth_macaddr_get failed\n");

    // Try to find an IPv4 address, defaulting to 127.0.0.1
    *ipv4_addr = RTE_BE32(RTE_IPV4_LOOPBACK);
    if (ifname)
    {
        struct ifaddrs *ifap = NULL;
        ret = getifaddrs(&ifap);
        if (ret != 0)
            rte_panic("getifaddrs failed\n");
        /* ifap points at a list of all addresses on the system. Try to find
         * one that's IPv4 with the right interface name.
         */
        for (struct ifaddrs *i = ifap; i; i = i->ifa_next)
        {
            if (strcmp(i->ifa_name, ifname) == 0
                && i->ifa_addr->sa_family == AF_INET)
            {
                *ipv4_addr = ((struct sockaddr_in *) i->ifa_addr)->sin_addr.s_addr;
                break;
            }
        }
        freeifaddrs(ifap);
    }
    return port_id;
}

static void append_to_mbuf(struct rte_mbuf *mbuf, const void *data, size_t size)
{
    void *ptr = rte_pktmbuf_append(mbuf, size);
    if (!ptr)
        rte_panic("rte_pktmbuf_append failed\n");
    memcpy(ptr, data, size);
}

struct prepare_mbuf_context
{
    struct rte_ether_addr mac;
    rte_be32_t ipv4_addr;
};

/* Fill in L2-L4 headers. The signature is to conform to the callback for
 * rte_mempool_obj_iter.
 */
static void prepare_mbuf(struct rte_mempool *mp, void *data, void *obj, unsigned obj_idx)
{
    struct rte_mbuf *mbuf = (struct rte_mbuf *) obj;
    const struct prepare_mbuf_context *ctx = data;

    struct rte_ether_hdr ether_hdr = {
        .dst_addr = MULTICAST_MAC,
        .src_addr = ctx->mac,
        .ether_type = RTE_BE16(RTE_ETHER_TYPE_IPV4)
    };

    struct rte_ipv4_hdr ipv4_hdr = {
        .version_ihl = 0x45,  // version 4, 20-byte header
        .total_length = rte_cpu_to_be_16(PAYLOAD_SIZE + sizeof(struct rte_udp_hdr) + sizeof(struct rte_ipv4_hdr)),
        .fragment_offset = RTE_BE16(0x4000),    // Don't-fragment
        .time_to_live = 4,
        .next_proto_id = IPPROTO_UDP,
        .src_addr = ctx->ipv4_addr,
        .dst_addr = MULTICAST_GROUP
    };

    struct rte_udp_hdr udp_hdr = {
        .src_port = SRC_PORT,
        .dst_port = MULTICAST_PORT,
        .dgram_len = rte_cpu_to_be_16(PAYLOAD_SIZE + sizeof(struct rte_udp_hdr)),
        .dgram_cksum = rte_ipv4_phdr_cksum(
            &ipv4_hdr,
            RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_UDP_CKSUM
        )
    };

    rte_pktmbuf_reset(mbuf);
    append_to_mbuf(mbuf, &ether_hdr, sizeof(ether_hdr));
    append_to_mbuf(mbuf, &ipv4_hdr, sizeof(ipv4_hdr));
    append_to_mbuf(mbuf, &udp_hdr, sizeof(udp_hdr));
}

static void setup_rx(uint16_t port_id, uint16_t nb_rx_desc)
{
    /* Don't actually want any RX, but can't set 0 queues. We also
     * have to give it a packet pool. The data_room_size is big to
     * ensure it exceeds the MTU.
     */
    struct rte_mempool *recv_mb_pool = rte_pktmbuf_pool_create("recv", 127, 0, 0, 16384, SOCKET_ID_ANY);
    if (!recv_mb_pool)
        rte_panic("rte_pktmbuf_pool_create failed\n");
    int ret = rte_eth_rx_queue_setup(port_id, 0, nb_rx_desc, SOCKET_ID_ANY, NULL, recv_mb_pool);
    if (ret != 0)
        rte_panic("rte_eth_rx_queue_setup failed\n");
}

int main(int argc, char **argv)
{
    int ret;

    ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_panic("Cannot init EAL\n");

    struct rte_ether_addr mac;
    rte_be32_t ipv4_addr;
    uint16_t port_id = choose_device(&mac, &ipv4_addr);

    /* Ignore any rx traffic which doesn't match a flow rule
     * (i.e., all of it, since none get set up).
     */
    ret = rte_flow_isolate(port_id, 1, NULL);
    if (ret != 0)
        rte_panic("rte_flow_isolate failed\n");

    struct rte_eth_conf eth_conf = {};
    eth_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_MULTI_SEGS;
    ret = rte_eth_dev_configure(port_id, 1, 1, &eth_conf);
    if (ret != 0)
        rte_panic("rte_eth_dev_configure failed\n");

    uint16_t nb_rx_desc = 1;
    uint16_t nb_tx_desc = 128;
    rte_eth_dev_adjust_nb_rx_tx_desc(port_id, &nb_rx_desc, &nb_tx_desc);

    // * 2 - 1 to hopefully get one less than a power of 2, which is apparently optimal
    int socket_id = rte_eth_dev_socket_id(port_id);
    struct rte_mempool *header_pool = rte_pktmbuf_pool_create(
        "headers", nb_tx_desc * 2 - 1,
        0, 0,
        RTE_MBUF_DEFAULT_DATAROOM,
        socket_id
    );
    if (!header_pool)
        rte_panic("rte_pktmbuf_pool_create failed (header_pool)\n");

    struct rte_mempool *payload_pool = NULL;
#if USE_EXTERNAL
    size_t ext_size = 2 * 1024 * 1024;
    void *ext = mmap(NULL, ext_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE | MAP_LOCKED, -1, 0);
    if (!ext)
        rte_panic("mmap failed");
    rte_extmem_register(ext, ext_size, NULL, 0, ext_size);

    const struct rte_pktmbuf_extmem ext_mem =
    {
        .buf_ptr = ext,
        .buf_iova = (rte_iova_t) ext,   // TODO: is this valid?
        .buf_len = ext_size,
        .elt_size = 4096
    };
    payload_pool = rte_pktmbuf_pool_create_extbuf(
        "payload", nb_tx_desc * 2 - 1, 0, 0,
        4096,
        socket_id,
        &ext_mem, 1);
#else
    payload_pool = rte_pktmbuf_pool_create(
        "payload", nb_tx_desc * 2 - 1, 0, 0,
        4096,
        socket_id
    );
#endif
    if (!payload_pool)
        rte_panic("rte_pktmbuf_pool_create failed (payload_pool)\n");
    struct prepare_mbuf_context ctx = {mac, ipv4_addr};
    rte_mempool_obj_iter(header_pool, prepare_mbuf, &ctx);

    ret = rte_eth_tx_queue_setup(port_id, 0, nb_tx_desc, socket_id, NULL);
    if (ret != 0)
        rte_panic("rte_eth_tx_queue_setup failed\n");

    setup_rx(port_id, nb_rx_desc);

    ret = rte_eth_dev_start(port_id);
    if (ret != 0)
        rte_panic("rte_eth_dev_start failed\n");

    for (int i = 0; i < 200; i++)
    {
        struct rte_mbuf *payload_mbuf = rte_pktmbuf_alloc(payload_pool);
        if (!payload_mbuf)
            rte_panic("rte_pktmbuf_alloc failed\n");
        struct rte_mbuf *header_mbuf = rte_pktmbuf_alloc(header_pool);
        if (!header_mbuf)
            rte_panic("rte_pktmbuf_alloc_bulk failed\n");

        /* Extend data pointer to encompass the pre-written header */
        rte_pktmbuf_append(
            header_mbuf,
            sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr));
        header_mbuf->ol_flags |= RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_UDP_CKSUM;
        header_mbuf->l2_len = sizeof(struct rte_ether_hdr);
        header_mbuf->l3_len = sizeof(struct rte_ipv4_hdr);
        /* Add the payload */
        rte_pktmbuf_append(payload_mbuf, PAYLOAD_SIZE);
        /* Chain the buffers */
        rte_pktmbuf_chain(header_mbuf, payload_mbuf);

        // Send the packet
        do
        {
            ret = rte_eth_tx_burst(port_id, 0, &header_mbuf, 1);
        } while (ret == 0);
        if (ret != 1)
            rte_panic("rte_eth_tx_burst failed\n");
    }

    rte_eth_dev_stop(port_id);
    // Should be more cleanup to free the memory pools
    rte_eal_cleanup();
    return 0;
}
