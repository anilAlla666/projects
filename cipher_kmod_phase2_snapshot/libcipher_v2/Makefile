# libcipher_v2 -- CUDA injection library for Phase 2 tenant identity flow.
# Depends only on libc + libpthread. No CUDA, no CUPTI, no libcuda.

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -fPIC -fvisibility=hidden
LDFLAGS ?= -shared -fPIC -lpthread

# cipher_ioctl.h lives with the kmod sources; share via -I rather than copy.
CIPHER_KMOD_DIR ?= /workspace/cipher_kmod

# Symbols we DO want exported (InitializeInjection variants need to be
# dlsym-able by the driver). Everything else stays hidden via the
# fvisibility default above.
EXPORTED := -fvisibility=default

TARGET := libcipher_v2.so
OBJS   := cipher_inject.o cipher_tenant.o

all: $(TARGET)

cipher_inject.o: cipher_inject.c cipher_v2_internal.h
	$(CC) $(CFLAGS) -fvisibility=default \
	      -I$(CIPHER_KMOD_DIR) -c -o $@ $<

cipher_tenant.o: cipher_tenant.c cipher_v2_internal.h $(CIPHER_KMOD_DIR)/cipher_ioctl.h
	$(CC) $(CFLAGS) -I$(CIPHER_KMOD_DIR) -c -o $@ $<

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
