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

device_info choose_device();

#endif // DPDK_COMMON_H
