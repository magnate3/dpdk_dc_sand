/**
 * @file    custom_ibverbs_tx.c
 *
 * @brief   Sample program that demonstrates how to transmit raw udp ethernet network data at high data rates using the 
 *          ibverbs library.
 * 
 * TODO: Give it in more details
 * 
 * @author  Gareth Callanan
 *          South African Radio Astronomy Observatory(SARAO)
 */

#include <infiniband/verbs.h>
#include <stdio.h>
#include <unistd.h>
/* Functions in this library are used for converting IP address strings to character arrays and for converting data 
 * between network and host byte order.
 */
#include <arpa/inet.h> 
#include <sys/time.h>   //For timing functions

#define SQ_NUM_DESC 2048 /* maximum number of sends waiting for completion - 2048 seems to be the maximum*/
#define NUM_WE_PER_POST_SEND 64
#define DESTINATION_IP_ADDRESS "10.100.18.7"
//Store MAC_ADDRESS as a sequence of bytes, this is the easiest to work with for a simple example as there is no simple string to mac address function like the inet_addr function for IP addresses.
#define DESTINATION_MAC_ADDRESS {0x1c,0x34,0xda,0x4b,0x93,0x92}
#define SOURCE_IP_ADDRESS "10.100.18.9"
#define SOURCE_MAC_ADDRESS {0x1c,0x34,0xda,0x54,0x99,0xbc}
#define UDP_PORT 7708
#define PAYLOAD_SIZE_BYTES 4096

struct __attribute__((__packed__)) network_packet {
    uint8_t ethernet_frame_dest_mac[6];
    uint8_t ethernet_frame_src_mac[6];
    uint16_t ethernet_frame_ether_type;
    
    uint8_t ip_packet_version_and_ihl;
    uint8_t ip_packet_dscp_and_ecn;
    uint16_t ip_packet_total_length;
    uint16_t ip_packet_identification;
    uint16_t ip_packet_flags_and_fragment_offset;
    uint8_t ip_packet_ttl;
    uint8_t ip_packet_protocol;
    uint16_t ip_packet_checksum;
    uint32_t ip_packet_src_address;
    uint32_t ip_packet_dest_address;

    uint16_t upd_datagram_src_port;
    uint16_t upd_datagram_dest_port;
    uint16_t upd_datagram_length;
    uint16_t upd_datagram_checksum;
    uint8_t udp_datagram_payload[PAYLOAD_SIZE_BYTES];
};


struct ibv_device * get_ibv_device_from_ip(uint8_t * u8PortNumber);
void populate_packet(struct network_packet * p_network_packet, struct ibv_context * p_context);


int main()
{
    struct ibv_device *ib_dev;
    struct ibv_context *context;
    struct ibv_pd *pd;
    int ret;
    uint8_t u8PortNum;

    printf("Network Packet Size: %ld\n", sizeof(struct network_packet));

    /* 1. Get correct device and physical port number from source IP address specified by SOURCE_IP_ADDRESS */
    ib_dev = get_ibv_device_from_ip(&u8PortNum);
    if(ib_dev == NULL){
        printf("No NIC with matching SOURCE_IP_ADDRESS found\n");
        exit(1);
    }

    /* 2. Get the device context */
    /* Get context to device. The context is needed for resource tracking and operations */
    context = ibv_open_device(ib_dev);
    if (!context)
    {
        printf("Couldn't get context for %s\n", ibv_get_device_name(ib_dev));
        exit(1);
    }

    /* 3. Allocate Protection Domain */
    /* Allocate a protection domain to group memory regions (MR) and rings */
    pd = ibv_alloc_pd(context);
    if (!pd)
    {
        printf("Couldn't allocate PD\n");
        exit(1);
    }

    /* 4. Create Complition Queue (CQ) */
    struct ibv_cq *cq;
    cq = ibv_create_cq(context, SQ_NUM_DESC, NULL, NULL, 0);
    if (!cq)
    {
        printf("Couldn't create CQ %d\n", errno);
        exit (1);
    }

    /* 5. Initialize QP */
    struct ibv_qp *qp;
    struct ibv_qp_init_attr qp_init_attr = {
        .qp_context = NULL,
        /* report send completion to cq */
        .send_cq = cq,
        .recv_cq = cq,
        .cap = {
            /* number of allowed outstanding sends without waiting for a completion - completions are explained
             * further down in the program.
             */
            .max_send_wr = SQ_NUM_DESC,
            /* maximum number of pointers in each descriptor */
            .max_send_sge = 1,
            /* if inline maximum of payload data in the descriptors themselves */
            .max_inline_data = 512,
            .max_recv_wr = 0
        },
        /* This flag specifies that we are constructing raw ethernet packets. */
        .qp_type = IBV_QPT_RAW_PACKET, 
    };

    /* 6. Create Queue Pair (QP) - Send Ring */
    qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp)
    {
        printf("Couldn't create RSS QP\n");
        exit(1);
    }

    /* 7. Initialize the QP (receive ring) and assign the correct physical port */
    struct ibv_qp_attr qp_attr;
    int qp_flags;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_flags = IBV_QP_STATE | IBV_QP_PORT;
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.port_num = u8PortNum; //I have never had this value equal to anything other than 1, I have a niggling concern that if it equals another number things will not work;
    ret = ibv_modify_qp(qp, &qp_attr, qp_flags);
    if (ret < 0)
    {
        printf("failed modify qp to init\n");
        exit(1);
    }
    memset(&qp_attr, 0, sizeof(qp_attr));

    /* 8. Move this ring to a "ready" state. Both the ready to send and receive states need to be entered. */
    /* 8.1. Move ring state to ready to receive */
    qp_flags = IBV_QP_STATE;
    qp_attr.qp_state = IBV_QPS_RTR;
    ret = ibv_modify_qp(qp, &qp_attr, qp_flags);
    if (ret < 0)
    {
        printf("failed modify qp to receive\n");
        exit(1);
    }

    /* 8.2. Move ring state to ready to send */
    qp_flags = IBV_QP_STATE;
    qp_attr.qp_state = IBV_QPS_RTS;
    ret = ibv_modify_qp(qp, &qp_attr, qp_flags);
    if (ret < 0)
    {
        printf("failed modify qp to receive\n");
        exit(1);
    }

    /* 9. Allocate and populate memory - this is user space memory that will be read by the NIC. This memory will need to contain raw data as well as ethernet frame/ip packet/udp datagram headers */
    /* 9.1 Allocate a space in memory to accomodate SQ_NUM_DESC packets. */
    uint64_t u64PacketBufferSize = sizeof(struct network_packet)*SQ_NUM_DESC;
    struct network_packet *psPacketBuffer = malloc(u64PacketBufferSize);
    if (!psPacketBuffer)
    {
        printf("Could not allocate memory\n");
        exit(1);   
    }

    /* 9.2 Populate the network headers of each packet in the buffer - The data fields of these packets are left blank */
    struct network_packet sSinglePacket;
    memset(&sSinglePacket,0x00,sizeof(struct network_packet));
    populate_packet(&sSinglePacket, context);
    for (size_t i = 0; i < SQ_NUM_DESC; i++)
    {
        memcpy(&psPacketBuffer[i], &sSinglePacket, sizeof(struct network_packet));
    }

    /* 10. Register the user memory so it can be accessed by the HW directly */
    struct ibv_mr *mr;
    mr = ibv_reg_mr(pd, psPacketBuffer, u64PacketBufferSize, IBV_ACCESS_LOCAL_WRITE);
    if (!mr)
    {
        printf("Couldn't register mr\n");
        exit(1);
    }

    /* 11. Configure all data structures that will be communicate information to the NIC
     *  There are two main structures used here. A work request(WR) and a scatter gather entry(SGE).
     *  a) An SGE describes the location and size of data to send. In our instance an SGE points to a single packet
     *  b) For sending, a WR is posted to the NIC and describes the transaction to be performed in the NIC. For this
     *  simple case a WR tells the NIC that a "send" operation takes place and provides a pointer to the SGE which 
     *  points to the data to be sent.
     *      i) Multiple WRs can be chained together as a linked list(wr[x].next points to next WR in list). The first 
     *         wr needs to be posted to the QP and the NIC will handle the entire WR list. This further reduces the load
     *         on the CPU. For this example. NUM_WE_PER_POST_SEND WRs are posted to the NIC at once. We have SQ_NUM_DESC WRs
     *         as we can have that many in the send queue at any one time - each needs to be unique so we can point to a 
     *         unique SGE 
     *      ii) A WR can point to multiple SGEs. This behaviour is not made very obvious in the documentation. I think 
     *         all the different SGEs will be combined into a single packet, but it could be transmitted as multiple
     *         packets. If this were the case, you could have the header info in a single SGE and the payload in a 
     *         different SGE which could be useful. It may be that this is not even supported for raw ethernet packet
     *         mode. Someone should test this out. All the examples I have seen have a single SGE per WR.
     */
    
    
    /* 11.1 Initialise all required structs */
    struct ibv_sge sg_entry[SQ_NUM_DESC];
    struct ibv_send_wr wr[SQ_NUM_DESC];
    memset(wr, 0, sizeof(wr[0])*SQ_NUM_DESC);

    /* 11.2 Link SGEs to WRs and link WRs together in a linked list */
    for (size_t i = 0; i < SQ_NUM_DESC; i++)
    {
        //Point the scatter gather entry to the correct packet
        sg_entry[i].addr = (uint64_t)&psPacketBuffer[i];
        sg_entry[i].length = sizeof(struct network_packet);
        sg_entry[i].lkey = mr->lkey;

        wr[i].num_sge = 1;
        wr[i].sg_list = &sg_entry[i]; //Point WR to SGE
        wr[i].next = &wr[i+1]; //Link WRs together 
        wr[i].opcode = IBV_WR_SEND;
        wr[i].send_flags = 0;
    }

    for (size_t i = NUM_WE_PER_POST_SEND; i <= SQ_NUM_DESC; i+=NUM_WE_PER_POST_SEND)
    {
        wr[i-1].next = NULL; //Last WR in list must not point to any further WRs.
    
        /* By setting this IBV_SEND_SIGNALED flag, the last WR will generate a work completion event once the packet has 
        * been transmitted. Explained in more detail below.
        */
        wr[i-1].send_flags |= IBV_SEND_SIGNALED;
    }  

    /* 12. Send Operation 
     * The send operation is very lightweight and simple. It requires calling ibv_post_send(). This function will tell
     * then NIC to transmit all packets in the completion list.
     *  
     * Only a limited number of WRs can queued on the NIC at any one time (see the ".max_send_wr = SQ_NUM_DESC" line 
     * in the ibv_qp_init_attr struct". In order to track this, ibverbs defines a Work Completion(WC). This is a struct
     * that can be returned by polling the NIC with the ibv_poll_cq command. In this example, I have configured a
     * completion event to occur at the end of of each WR list by setting the wr send flag in the last element of the 
     * list. See line: "wr[NUM_WE_PER_POST_SEND-1].send_flags |= IBV_SEND_SIGNALED" above.
     * 
     * I count the packets and add this packet index in the first 8 bytes of the UDP payload. The receiver can then use
     * this packet index number to detrmine if the packet have been dropped or received out of order.
     */

    uint64_t u64NumPostedTotal = 0;
    uint64_t u64NumCompletedTotal = 0;
    int iNumCompletedByWR;
    struct ibv_wc wc;
    struct ibv_send_wr * bad_wr = NULL;

    //For measuring the data rate
    printf("\nStarting Transmission:\n\n");

    struct timeval sTimerStartTime;
    struct timeval sInitialStartTime;
    struct timeval sCurrentTime;

    /* These two variables are used for tracking the number of post sends at the start of the timing window and at the
    end - only use for timing*/
    uint64_t u64StartPostSendCount = 0;
    uint64_t u64CurrentPostSendCount;

    uint64_t u64PacketIndex = 0;
    uint64_t u64NextWRPostSendIndex = 0;

    gettimeofday(&sInitialStartTime,NULL);
    gettimeofday(&sTimerStartTime,NULL);

    while(1) 
    {

        /* This inline flag was part of the sample program I based this on. I am not sure how to use if effectivly.
         * I am leaving it here as a reminder to investigate further. */
        //wr.send_flags = IBV_SEND_INLINE;
    
        /* 12.1 Track which set of WRs is being transmitted next and update the count on these packets */
        for (size_t i = 0; i < NUM_WE_PER_POST_SEND; i++)
        {
            *(uint64_t*) &psPacketBuffer[u64NextWRPostSendIndex * NUM_WE_PER_POST_SEND + i].udp_datagram_payload = u64PacketIndex;
            u64PacketIndex++;
        }
        u64NextWRPostSendIndex = (u64NextWRPostSendIndex + 1) % (SQ_NUM_DESC / NUM_WE_PER_POST_SEND);

        /* 12.2 Push WRs to hardware */
        ret = ibv_post_send(qp, &wr[u64NextWRPostSendIndex*NUM_WE_PER_POST_SEND], &bad_wr);
        if (ret != 0 /*|| bad_wr != NULL*/) 
        {
            printf("failed in post send. Errno: %d\n",ret);
            exit(1);
        }
        u64NumPostedTotal++;
        
        /* 12.3 poll for completion after half ring is posted */
        if (u64NumPostedTotal - u64NumCompletedTotal > 0)
        {
            iNumCompletedByWR = 0;
            /* ibv_poll_cq is non blocking and will return zero most of the time. If the send queue is not full this is
             * not an issue, but if the send queue is full, ibv_post_send will return error code 12. As such if the
             * queue is full, keep polling continously until it is no longer full.
             */
            do{
                iNumCompletedByWR = ibv_poll_cq(cq, 1, &wc);
                u64NumCompletedTotal+=iNumCompletedByWR;
                if (iNumCompletedByWR > 0) 
                {   
                    //Here for debug purposes, empty otherwise
                    //printf("%d %ld %d %ld %d\n",iNumCompletedByWR,wc.wr_id,wc.opcode,wc.status, u64NumPostedTotal - u64NumCompletedTotal);
                }
                else if (iNumCompletedByWR < 0) 
                {
                    printf("Polling error\n");
                    exit(1);
                }
            }while(u64NumPostedTotal - u64NumCompletedTotal >= SQ_NUM_DESC/NUM_WE_PER_POST_SEND);
        }

        //Measure time and if a second has passed print the data rate to screen.
        gettimeofday(&sCurrentTime,NULL);
        double dTimeDifference = (double)sCurrentTime.tv_sec + ((double)sCurrentTime.tv_usec)/1000000.0
                            - (double)sTimerStartTime.tv_sec - ((double)sTimerStartTime.tv_usec)/1000000.0;
        if(dTimeDifference > 2){
            //Calculate data rate
            u64CurrentPostSendCount = u64NumCompletedTotal;
            double dDataTransferred_Gb = (u64CurrentPostSendCount - u64StartPostSendCount) * NUM_WE_PER_POST_SEND * sizeof(struct network_packet)/1000000000 * 8;
            double dDataRate_Gbps = dDataTransferred_Gb/dTimeDifference;
            double dTotalDataTransferred_GB = u64NumCompletedTotal * NUM_WE_PER_POST_SEND * sizeof(struct network_packet)/1000000000;
            double dRuntime_s = (double)sCurrentTime.tv_sec + ((double)sCurrentTime.tv_usec)/1000000.0
                            - (double)sInitialStartTime.tv_sec - ((double)sInitialStartTime.tv_usec)/1000000.0;
            printf("\rRunning Time: %.2fs. Total Transmitted %.3f GB. Current Data Rate: %.3f Gbps",dRuntime_s,dTotalDataTransferred_GB,dDataRate_Gbps);
            fflush(stdout);

            //Set timer up for next second
            u64StartPostSendCount = u64CurrentPostSendCount;
            sTimerStartTime = sCurrentTime;
        }

    }

    printf("We are done\n");
    return 0;
}


struct ibv_device * get_ibv_device_from_ip(uint8_t * u8PortIndex){
    
    struct ibv_device **dev_list;
    struct ibv_device *ib_dev;
    struct ibv_context *p_context;
    int iNumDevices;
    uint8_t u8DeviceFound = 0;

    /* 1. Store the source IP address as four octets for ease of comparison */
    uint8_t pu8SourceAddrOctets[4];
    ((uint32_t *) pu8SourceAddrOctets)[0] = inet_addr(SOURCE_IP_ADDRESS);

    /* 2. Get the list of offload capable devices */
    dev_list = ibv_get_device_list(&iNumDevices);
    if (!dev_list)
    {
        printf("Failed to get devices list");
        exit(1);
    }
    
    /* 3. Iterate through all offload capable devices to find the one with the correct IP address*/
    for (size_t i = 0; i < iNumDevices; i++)
    {   
        ib_dev = dev_list[i];
        printf("RDMA device[%ld]: name=%s\n", i, ibv_get_device_name(ib_dev));
        if (!ib_dev)
        {
            printf("IB device not found\n");
            exit(1);
        }
        
        /* 3.1 The device context is required by the ibv_query_gid() function */
        p_context = ibv_open_device(ib_dev);
        if (!p_context)
        {
            printf("Couldn't get context for %s\n", ibv_get_device_name(ib_dev));
            exit(1);
        }

        /* 3.2 Iterate through all the ports of each device*/
        union ibv_gid gid;
        *u8PortIndex = 1;
        while(1)
        {
            /* 3.2.1 Get the port GID*/
            //Not sure why but the second argument works when set to 2 for my test configuration, but 1 for others. If your IP address is not detected, try change the value to 1.
            int rc = ibv_query_gid(p_context, *u8PortIndex, 2, &gid);
            if (rc) 
            {
                break;
            }
            printf("\tPhysical Port: %d\n",*u8PortIndex);
            printf("\t\tGID: GID Prefix: %d %d %d %d %d %d %d %d\n",(uint32_t)gid.raw[0], (uint32_t)gid.raw[1], (uint32_t)gid.raw[2], (uint32_t)gid.raw[3], (uint32_t)gid.raw[4], (uint32_t)gid.raw[5], (uint32_t)gid.raw[6], (uint32_t)gid.raw[7]);
            printf("\t\tGID: Subnet Prefix: %d %d %d %d %d %d %d %d\n", (uint32_t)gid.raw[8], (uint32_t)gid.raw[9], (uint32_t)gid.raw[10], (uint32_t)gid.raw[11], (uint32_t)gid.raw[12], (uint32_t)gid.raw[13], (uint32_t)gid.raw[14] , (uint32_t)gid.raw[15]);
            printf("\t\tIP Address From GID: %d.%d.%d.%d\n",(uint32_t)gid.raw[12], (uint32_t)gid.raw[13], (uint32_t)gid.raw[14] , (uint32_t)gid.raw[15]);

            /* 3.2.2 Compare the fields in the GID that correspond to IP address with the expected IP address */
            if(pu8SourceAddrOctets[0] == gid.raw[12] && pu8SourceAddrOctets[1] == gid.raw[13] && pu8SourceAddrOctets[2] == gid.raw[14] && pu8SourceAddrOctets[3] == gid.raw[15])
            {
                u8DeviceFound = 1;
                break;
            }

            *u8PortIndex = *u8PortIndex + 1;
        }

        /* 3.3 Cleanup */
        ibv_close_device(p_context);
        if(u8DeviceFound){
            break;
        }

    }

    /* 4. Set pointer to NULL if no device is found for a safe exit condition */
    if(u8DeviceFound == 0){
        *u8PortIndex = 0;
        ib_dev = NULL;
    }

    /* 5. Cleanup */
    ibv_free_device_list(dev_list);

    /* 6. Return device pointer*/
    return ib_dev;
}

void populate_packet(struct network_packet * p_network_packet, struct ibv_context * p_context){
    
    //Transport Layer(L4)
    p_network_packet->upd_datagram_src_port = 0; //Keep to zero if no reply expected.
    p_network_packet->upd_datagram_dest_port = htons(UDP_PORT);
    
    /* The UDP checksum has been set to zero as it is quite expensive to calculate and is not checked by any of
     * MeerKATs systems. If this is required, there are methods to offload this calculation to the NIC. From what I can
     * tell, this is only supported on MLNX_OFED 5 and above. This program was tested on MLNX_OFED version 4 and as such
     * I have not looked too deeply into offloads. If you want to implement it yourself, see the IBV_SEND_IP_CSUM 
     * flag described here: https://manpages.debian.org/testing/libibverbs-dev/ibv_post_send.3.en.html. 
     */
    p_network_packet->upd_datagram_checksum = 0; 

    //The length of UDP datagram includes UDP header and payload
    uint16_t u16UDPDatagramLengthBytes= PAYLOAD_SIZE_BYTES
        + sizeof(p_network_packet->upd_datagram_src_port)
        + sizeof(p_network_packet->upd_datagram_dest_port)
        + sizeof(p_network_packet->upd_datagram_checksum)
        + sizeof(p_network_packet->upd_datagram_length);
    
    p_network_packet->upd_datagram_length = htons(u16UDPDatagramLengthBytes);
    //IP Layer(L3)

    //This value is hardcoded - have not bothered to look into it
    p_network_packet->ip_packet_version_and_ihl = 0x45;
    
    //These values allow for differentiating type of service and congestion level - I think this is not used in the MeerKAT network, so it is just left at 0
    p_network_packet->ip_packet_dscp_and_ecn = 0x0;

    //This is the packet length - it includes the IP header and data while excluding the ethernet frame fields.
    uint16_t u16IPPacketLengthBytes = sizeof(*p_network_packet) 
            - sizeof(p_network_packet->ethernet_frame_dest_mac) 
            - sizeof(p_network_packet->ethernet_frame_src_mac)
            - sizeof(p_network_packet->ethernet_frame_ether_type);
    p_network_packet->ip_packet_total_length = htons(u16IPPacketLengthBytes);

    //If an IP packet is fragmented during transmission, this field contains a unique number identifying the original packet when packets are reassembled.
    p_network_packet->ip_packet_identification = 0;

    //This specifies if a packet can be fragmented and what the offset of the current fragment is. We set a flag to disable fragmentation
    p_network_packet->ip_packet_flags_and_fragment_offset = htons(0x4000);

    //TTL is well explained - google it. Set to eight for now, but that was more a guess. May need to be reworked in the future
    p_network_packet->ip_packet_ttl = 8;

    //17 represents the UDP protocol
    p_network_packet->ip_packet_protocol = 17;

    //Set IP addresses
    p_network_packet->ip_packet_dest_address = inet_addr(DESTINATION_IP_ADDRESS);
    p_network_packet->ip_packet_src_address = inet_addr(SOURCE_IP_ADDRESS);

    //Calculating the checksum - break the header into 16 bit chunks. Sum these 16 bit chunks together and then 1 compliment them.
    p_network_packet->ip_packet_checksum = 0; //Must start off as zero for the calculation
    
    //1. Array of 16 bit chunks
    uint16_t * pu16IPHeader = (uint16_t *) &p_network_packet->ip_packet_version_and_ihl;

    //2.1 16-bit sum of data - we store it as a 32 bit number as the carry values need to be used.
    uint32_t u32Sum = 0;
    for (size_t i = 0; i < 10; i++)
    {
        u32Sum += ntohs(pu16IPHeader[i]);//Remember network byte order
    }
    //2.2 Compensate for carry - every time a carry occurs, add one to the u32Sum. Can do this at the end as follows:
    //This has not actually been tested yet
    //At first glance this could be an if statement, however if adding all the carry bits causes an additional carry, then this step needs to occur again, this is why a while loop is necessary. 
    while (u32Sum > 0xffff){
        u32Sum = (u32Sum & 0xffff) + (u32Sum >> 16);
    }

    //3. 1s compliment the data
    uint16_t u16SumComplimented = ~(uint16_t)u32Sum;

    //4. Store checksum in packet.
    p_network_packet->ip_packet_checksum = htons(u16SumComplimented);

    //Ethernet Layer(L2)
    uint8_t pu8DestMacAddress[6] = DESTINATION_MAC_ADDRESS;
    uint8_t pu8SrcMacAddress[6] = SOURCE_MAC_ADDRESS;
    for (size_t i = 0; i < 6; i++)
    {   
        p_network_packet->ethernet_frame_dest_mac[i] = pu8DestMacAddress[i];
        p_network_packet->ethernet_frame_src_mac[i] = pu8SrcMacAddress[i];
    }
    p_network_packet->ethernet_frame_ether_type = htons(0x0800);
    
    uint8_t * p_temp = (uint8_t*) & p_network_packet->ip_packet_src_address;
    printf("Source IP: %d.%d.%d.%d\n",(int32_t)p_temp[0],(int32_t)p_temp[1],(int32_t)p_temp[2],(int32_t)p_temp[3]);

    p_temp = (uint8_t*) & p_network_packet->ip_packet_dest_address;
    printf("Destination IP: %d.%d.%d.%d\n",(int32_t)p_temp[0],(int32_t)p_temp[1],(int32_t)p_temp[2],(int32_t)p_temp[3]);
    
    p_temp = (uint8_t*) & p_network_packet->ethernet_frame_dest_mac;
    printf("Destination MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",(int32_t)p_temp[0],(int32_t)p_temp[1],(int32_t)p_temp[2],(int32_t)p_temp[3],(int32_t)p_temp[4],(int32_t)p_temp[5]);
    
    p_temp = (uint8_t*) & p_network_packet->ethernet_frame_src_mac;
    printf("Source MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",(int32_t)p_temp[0],(int32_t)p_temp[1],(int32_t)p_temp[2],(int32_t)p_temp[3],(int32_t)p_temp[4],(int32_t)p_temp[5]);
    
    printf("Packet Length excluding frame: %d bytes\n",(int32_t)u16IPPacketLengthBytes);
    //Ethernet Layer
    //p_network_packet->ethernet_frame_dest_mac = 
    //p_network_packet->ethernet_frame_src_mac = 

    // getifaddrs (&ifap);
    // for (ifa = ifap; ifa; ifa = ifa->ifa_next) {
    //     if (ifa->ifa_addr && ifa->ifa_addr->sa_family==AF_INET) {
    //         sa = (struct sockaddr_in *) ifa->ifa_addr;
    //         addr = inet_ntoa(sa->sin_addr);
    //         uint8_t addr_octets[4];
    //         ((uint32_t *) addr_octets)[0] = *(uint32_t*) &sa->sin_addr;
            
    //         if(addr_octets[0] == gid.raw[12] && addr_octets[1] == gid.raw[13] && addr_octets[2] == gid.raw[14] && addr_octets[3] == gid.raw[15])
    //         {
    //             printf("Interface: %s\tAddress: %s\n", ifa->ifa_name, addr);
    //             struct sockaddr_ll *ll = (struct sockaddr_ll *) ifa->ifa_addr;
    //             if (ll->sll_hatype == ARPHRD_ETHER && ll->sll_halen == 6)
    //             {
    //                 memcpy(&p_network_packet->ethernet_frame_src_mac, ll->sll_addr, 6);
    //             }
    //             break;
    //         }
    //     }
    // }

    // p_network_packet->ip_packet_src_address = inet_addr(addr);
    // printf("src ip address: %d\n", p_network_packet->ip_packet_src_address);
    // printf("src mac address: %d:%d:%d:%d:%d:%d\n",
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[0],
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[1],
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[2],
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[3],
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[4],
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[5]);

    // freeifaddrs(ifap);
}
