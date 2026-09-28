# CIPHER kernel module — out-of-tree build wrapper.
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

default: modules

modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

modules_install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install

.PHONY: default modules clean modules_install
