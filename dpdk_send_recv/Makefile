CXX = g++ -std=c++17
CFLAGS = -Wall -g $(shell pkg-config --cflags libdpdk)
LIBS = $(shell pkg-config --libs libdpdk)
SOURCES = $(wildcard *.cpp)
TARGETS = dpdk_send  # dpdk_recv

all: $(TARGETS)

dpdk_send: dpdk_send.o dpdk_common.o
	$(CXX) -o $@ $^ $(LIBS)

dpdk_recv: dpdk_recv.o dpdk_common.o
	$(CXX) -o $@ $^ $(LIBS)

%.o: %.cpp *.h
	$(CXX) -c $< $(CFLAGS)

clean:
	rm -f $(TARGETS) *.o
