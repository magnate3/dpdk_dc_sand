CXX = g++
CFLAGS = -Wall -g $(shell pkg-config --cflags libdpdk)
LIBS = $(shell pkg-config --libs libdpdk)
SOURCES = $(wildcard *.cpp)
TARGETS = $(patsubst %.cpp, %, $(SOURCES))

all: $(TARGETS)

%: %.cpp Makefile
	$(CXX) -o $@ $< $(CFLAGS) $(LIBS)

clean:
	rm -f $(TARGETS) *.o
