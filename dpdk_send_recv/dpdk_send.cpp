/* Example application using DPDK to send arbitrary data on a network.
 * Written to learn about DPDK - not meant to be an example of good code.
 *
 * It roughly emulates a spead2 use case by using external buffers. A few
 * largish chunks are allocated, and updated and transmitted in turn.
 * Before a buffer can be reused, we check that it has finished transmitting.
 */

#include <iostream>
#include <cstring>
#include <cstdint>
#include <memory>
#include <vector>
#include <deque>
#include <stdexcept>
#include <sys/mman.h>

#include <rte_byteorder.h>
#include <rte_debug.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_flow.h>
#include <rte_ip.h>
#include <rte_pause.h>

#include "dpdk_common.h"

static constexpr int CHUNK_SIZE = 4 * 1024 * 1024;
static constexpr int PAYLOAD_SIZE = 4096;
static constexpr int N_CHUNKS = 2;

class munmap_deleter
{
private:
    std::size_t size;

public:
    explicit munmap_deleter(std::size_t size) : size(size) {}
    void operator()(void *ptr) const
    {
        munmap(ptr, size);
    }
};

template<typename T>
static std::unique_ptr<T[], munmap_deleter> make_unique_huge(std::size_t elements)
{
    std::size_t size = sizeof(T) * elements;
    void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB | MAP_LOCKED,
                     -1, 0);
    if (ptr == MAP_FAILED)
        throw std::bad_alloc();
    new(ptr) T[elements];
    return std::unique_ptr<T[], munmap_deleter>((T *) ptr, munmap_deleter(size));
}

class chunk
{
public:
    std::unique_ptr<std::uint64_t[], munmap_deleter> data;
    bool active = false;  // in use by HW
    rte_mbuf_ext_shared_info shared_info;
    rte_device *device;

    static void set_inactive(void *addr, void *opaque)
    {
        *(bool *) opaque = false;
    }

    explicit chunk(rte_device *device) :
        data(make_unique_huge<std::uint64_t>(CHUNK_SIZE / sizeof(data[0]))),
        device(device)
    {
        shared_info.free_cb = &chunk::set_inactive;
        shared_info.fcb_opaque = &active;
        const int PGSZ = 2 * 1024 * 1024;
        assert(((uintptr_t) data.get()) % PGSZ == 0);
        assert(CHUNK_SIZE % PGSZ == 0);
#if 0
        int pages = CHUNK_SIZE / PGSZ;
        std::vector<rte_iova_t> iova_addrs(pages);
        for (int i = 0; i < pages; i++)
            iova_addrs[i] = (rte_iova_t) (start + i * PGSZ);
        int ret = rte_extmem_register(start, end - start, iova_addrs.data(), pages, PGSZ);
#else
        int ret = rte_extmem_register(data.get(), CHUNK_SIZE, NULL, 0, PGSZ);
#endif
        if (ret != 0)
            rte_panic("rte_extmem_register failed");
        ret = rte_dev_dma_map(device, data.get(), (uintptr_t) data.get(), CHUNK_SIZE);
        if (ret != 0)
            rte_panic("rte_dev_dma_map failed");
    }

    ~chunk()
    {
        if (data)
        {
            rte_dev_dma_unmap(device, data.get(), (uintptr_t) data.get(), CHUNK_SIZE);
            rte_extmem_unregister(data.get(), CHUNK_SIZE);
        }
    }
};

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
        .src_addr = ctx.info->ipv4_addr,
        .dst_addr = MULTICAST_GROUP
    };

    // TODO: get a valid src port number from the kernel?
    rte_udp_hdr udp_hdr = {
        .src_port = rte_cpu_to_be_16(1234),
        .dst_port = MULTICAST_PORT,
        .dgram_len = rte_cpu_to_be_16(ctx.payload_size + sizeof(rte_udp_hdr)),
        .dgram_cksum = rte_ipv4_phdr_cksum(
            &ipv4_hdr,
            RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_UDP_CKSUM
        )
    };

    rte_pktmbuf_reset(mbuf);
    mbuf->data_off = 0;  // don't need (or have space for) the headroom
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
    std::cout << "Tx offload caps: " << std::hex << info.dev_info.tx_offload_capa << std::dec << '\n';
    std::cout << "Tx queue offload caps: " << std::hex << info.dev_info.tx_queue_offload_capa << std::dec << '\n';
    std::cout << "Max Tx segments: " << info.dev_info.tx_desc_lim.nb_seg_max << '\n';

    /* Ignore any rx traffic which doesn't match a flow rule
     * (i.e., all of it, since none get set up).
     */
    ret = rte_flow_isolate(info.port_id, 1, NULL);
    if (ret != 0)
        rte_panic("rte_flow_isolate failed\n");

    rte_eth_conf eth_conf = {};
    eth_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_MULTI_SEGS;
    ret = rte_eth_dev_configure(info.port_id, 1, 1, &eth_conf);
    if (ret != 0)
        rte_panic("rte_eth_dev_configure failed\n");

    uint16_t nb_rx_desc = 1;
    uint16_t nb_tx_desc = 128;
    rte_eth_dev_adjust_nb_rx_tx_desc(info.port_id, &nb_rx_desc, &nb_tx_desc);

    // * 2 - 1 to hopefully get one less than a power of 2, which is apparently optimal
    int socket_id = rte_eth_dev_socket_id(info.port_id);
    rte_mempool *header_pool = rte_pktmbuf_pool_create(
        "headers", nb_tx_desc * 2 - 1,
        0, 0,
        sizeof(rte_ether_hdr) + sizeof(rte_ipv4_hdr) + sizeof(rte_udp_hdr),
        socket_id
    );
    if (!header_pool)
        rte_panic("rte_pktmbuf_pool_create failed (header_pool)\n");
    rte_mempool *extbuf_pool = rte_pktmbuf_pool_create(
        "extbufs", nb_tx_desc * 2 - 1,
        0, 0,
        0,  // TODO: can size legally be zero?
        socket_id
    );
    if (!extbuf_pool)
        rte_panic("rte_pktmbuf_pool_create failed (extbuf_pool)\n");
    prepare_mbuf_context ctx = {&info, PAYLOAD_SIZE};
    rte_mempool_obj_iter(header_pool, prepare_mbuf, &ctx);

    ret = rte_eth_tx_queue_setup(info.port_id, 0, nb_tx_desc, socket_id, NULL);
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

    std::deque<chunk> chunks;
    for (int i = 0; i < N_CHUNKS; i++)
        chunks.emplace_back(info.dev_info.device);
    for (std::uint64_t chunk_id = 0; ; chunk_id++)
    {
        chunk &cur_chunk = chunks[chunk_id % N_CHUNKS];
        constexpr int num_packets = CHUNK_SIZE / PAYLOAD_SIZE;
        constexpr int max_burst = 32;
        static_assert(num_packets % max_burst == 0, "max_burst must currently divide into packet count");
        // Wait for previous use of the buffer (if any) to finish
        while (cur_chunk.active)
        {
            rte_pause();
            rte_eth_tx_done_cleanup(info.port_id, 0, 0);
            //std::cerr << rte_mbuf_ext_refcnt_read(&cur_chunk.shared_info) << '\n';
        }
        cur_chunk.active = true;
        /* Set the refcount for the full number of packets we're going to
         * attach to it.
         */
        rte_mbuf_ext_refcnt_set(&cur_chunk.shared_info, num_packets);
        for (int i = 0; i < num_packets; i += max_burst)
        {
            rte_mbuf *ext_mbufs[max_burst];
            ret = rte_pktmbuf_alloc_bulk(extbuf_pool, ext_mbufs, max_burst);
            if (ret != 0)
                rte_panic("rte_pktmbuf_alloc_bulk failed\n");

            rte_mbuf *header_mbufs[max_burst];
            ret = rte_pktmbuf_alloc_bulk(header_pool, header_mbufs, max_burst);
            if (ret != 0)
                rte_panic("rte_pktmbuf_alloc_bulk failed\n");

            for (int j = 0; j < max_burst; j++)
            {
                int n = i + j;
                std::size_t offset = std::size_t(PAYLOAD_SIZE) * n;
                header_mbufs[j]->data_off = 0;  // don't need (or have space for) the headroom
                /* Extend data pointer to encompass the pre-written headers */
                rte_pktmbuf_append(header_mbufs[j], sizeof(rte_ether_hdr) + sizeof(rte_ipv4_hdr) + sizeof(rte_udp_hdr));
                header_mbufs[j]->ol_flags |= RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_UDP_CKSUM;
                header_mbufs[j]->l2_len = sizeof(rte_ether_hdr);
                header_mbufs[j]->l3_len = sizeof(rte_ipv4_hdr);
                /* Add the payload */
                char *payload_ptr = offset + (char *) cur_chunk.data.get();
                // TODO: this just assumes IOVA == VA
                rte_pktmbuf_attach_extbuf(ext_mbufs[j], payload_ptr, (uintptr_t) payload_ptr, PAYLOAD_SIZE, &cur_chunk.shared_info);
                rte_pktmbuf_append(ext_mbufs[j], PAYLOAD_SIZE);
                /* Chain the buffers */
                rte_pktmbuf_chain(header_mbufs[j], ext_mbufs[j]);
            }
            // Send the packets. If the queue is full, try again.
            int rem = max_burst;
            rte_mbuf **next = header_mbufs;
            while (rem > 0)
            {
                ret = rte_eth_tx_burst(info.port_id, 0, next, rem);
                rem -= ret;
                next += ret;
            }
        }
        std::cerr << ".";
    }

    rte_eal_cleanup();
    return 0;
}
