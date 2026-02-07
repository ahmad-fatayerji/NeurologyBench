CC = gcc
CFLAGS = -O3 -lm -pthread
TARGET = net
BENCH = benchmark

all: $(TARGET) $(BENCH)

$(TARGET): net.c
	$(CC) $^ $(CFLAGS) -o $@

$(BENCH): benchmark.c net.c
	$(CC) $^ $(CFLAGS) -DNET_NO_MAIN -o $@

clean:
	rm -f $(TARGET) $(BENCH)

.PHONY: all clean

