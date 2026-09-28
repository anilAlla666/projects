# libcipher_v2 -- CUDA injection library.
# Phase 2: tenant identity flow (libc + libpthread only).
# Phase 3 Task 5: CUPTI kernel-launch callback (adds libcupti + libcuda).

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -fPIC -fvisibility=hidden
LDFLAGS ?= -shared -fPIC
# Libraries must come AFTER object files for --as-needed (the linker
# default on Ubuntu) to correctly emit DT_NEEDED entries.
LIBS    := -lpthread -lcupti -lcuda

# cipher_ioctl.h lives with the kmod sources; share via -I rather than copy.
CIPHER_KMOD_DIR ?= /home/ubuntu/cipher_kmod

# Symbols we DO want exported (InitializeInjection variants need to be
# dlsym-able by the driver). Everything else stays hidden via the
# fvisibility default above.
EXPORTED := -fvisibility=default

TARGET := libcipher_v2.so
OBJS   := cipher_inject.o cipher_tenant.o cipher_cupti.o

all: $(TARGET)

cipher_inject.o: cipher_inject.c cipher_v2_internal.h
	$(CC) $(CFLAGS) -fvisibility=default \
	      -I$(CIPHER_KMOD_DIR) -c -o $@ $<

cipher_tenant.o: cipher_tenant.c cipher_v2_internal.h $(CIPHER_KMOD_DIR)/cipher_ioctl.h
	$(CC) $(CFLAGS) -I$(CIPHER_KMOD_DIR) -c -o $@ $<

cipher_cupti.o: cipher_cupti.c cipher_v2_internal.h $(CIPHER_KMOD_DIR)/cipher_ioctl.h
	$(CC) $(CFLAGS) -I$(CIPHER_KMOD_DIR) -c -o $@ $<

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
