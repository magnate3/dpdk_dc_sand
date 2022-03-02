#ifndef DPDK_COMMON_H
#define DPDK_COMMON_H

#include <cstdint>
#include <string>

#include <rte_byteorder.h>
#include <rte_ether.h>
#include <rte_ethdev.h>

struct device_info
{
    std::uint16_t port_id;      // DPDK port number
    rte_eth_dev_info dev_info;  // DPDK-provided information
    std::string ifname;         // empty if it doesn't correspond to a hardware device
    rte_ether_addr mac;
    rte_be32_t ipv4_addr;       // 127.0.0.1 if no associated device address was found
};

// Pick the first enabled device and return information about it
device_info choose_device();

static constexpr rte_ether_addr MULTICAST_MAC = {{0x01, 0x00, 0x5E, 0x66, 0x11, 0x12}};
static constexpr rte_be32_t MULTICAST_GROUP = RTE_BE32(RTE_IPV4(239, 102, 17, 18));

#endif // DPDK_COMMON_H
