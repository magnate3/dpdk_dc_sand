#include "dpdk_common.h"

#include <net/if.h>
#include <ifaddrs.h>
#include <rte_debug.h>
#include <rte_ethdev.h>

device_info choose_device()
{
    bool found = false;
    int ret;
    device_info info{};
    std::uint16_t port_id;
    RTE_ETH_FOREACH_DEV(port_id)
    {
        ret = rte_eth_dev_info_get(port_id, &info.dev_info);
        if (ret != 0)
            rte_panic("rte_eth_dev_info_get failed\n");
        // If it corresponds to a kernel interface, we can get the name
        if (info.dev_info.if_index > 0)
        {
            char ifname_storage[IF_NAMESIZE];
            const char *ifname = if_indextoname(info.dev_info.if_index, ifname_storage);
            if (ifname)
                info.ifname = ifname;
        }
        info.port_id = port_id;
        found = true;
        break;
    }
    if (!found)
        rte_panic("no devices found\n");

    // Get the MAC address
    ret = rte_eth_macaddr_get(info.port_id, &info.mac);
    if (ret != 0)
        rte_panic("rte_eth_macaddr_get failed\n");

    // Try to find an IPv4 address, defaulting to 127.0.0.1
    info.ipv4_addr = RTE_BE32(RTE_IPV4_LOOPBACK);
    if (!info.ifname.empty())
    {
        ifaddrs *ifap = NULL;
        ret = getifaddrs(&ifap);
        if (ret != 0)
            rte_panic("getifaddrs failed\n");
        /* ifap points at a list of all addresses on the system. Try to find
         * one that's IPv4 with the right interface name.
         */
        for (ifaddrs *i = ifap; i; i = i->ifa_next)
        {
            if (std::string_view(i->ifa_name) == info.ifname
                && i->ifa_addr->sa_family == AF_INET)
            {
                info.ipv4_addr = ((struct sockaddr_in *) i->ifa_addr)->sin_addr.s_addr;
                break;
            }
        }
        freeifaddrs(ifap);
    }

    return info;
}
