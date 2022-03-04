CXX = g++ -std=c++17
CC = gcc
CFLAGS = -Wall -g $(shell pkg-config --cflags libdpdk)
LIBS = $(shell pkg-config --libs libdpdk)
TARGETS = dpdk_send dpdk_recv dpdk_send_demo

all: $(TARGETS)

dpdk_send: dpdk_send.o dpdk_common.o
	$(CXX) -o $@ $^ $(LIBS)

dpdk_recv: dpdk_recv.o dpdk_common.o
	$(CXX) -o $@ $^ $(LIBS)

dpdk_send_demo: dpdk_send_demo.o
	$(CC) -o $@ $^ $(LIBS)

%.o: %.cpp *.h
	$(CXX) -c $< $(CFLAGS)

dpdk_send_demo.o: dpdk_send_demo.c *.h
	$(CC) -c $< $(CFLAGS)

clean:
	rm -f $(TARGETS) *.o
