CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I/home/ubuntu/cipher_kmod
LDFLAGS = -lnvidia-ml

cipher-gpustate: cipher_gpustate.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f cipher-gpustate

.PHONY: clean
