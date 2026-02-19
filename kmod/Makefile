# SPDX-License-Identifier: GPL-2.0-only
#
# gpu-nvme-direct kernel module — wrapper Makefile
#
# Usage:
#   make                    Build the module against the running kernel
#   make KDIR=/path/to/src  Build against a specific kernel source tree
#   make clean              Remove build artifacts
#   make install            Install the module via modules_install

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

# Path to NVIDIA driver source (for nv-p2p.h)
# Set NVIDIA_SRC to your nvidia driver source, e.g.:
#   make NVIDIA_SRC=/usr/src/nvidia-550.54.14
NVIDIA_SRC ?=

# Export for Kbuild
export NVIDIA_SRC

.PHONY: all modules clean install help

all: modules

modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a

help:
	@echo "gpu-nvme-direct kernel module"
	@echo ""
	@echo "Targets:"
	@echo "  all / modules  - Build the gpunvme.ko module"
	@echo "  clean          - Remove build artifacts"
	@echo "  install        - Install module and update depmod"
	@echo ""
	@echo "Variables:"
	@echo "  KDIR=<path>        Kernel build directory (default: running kernel)"
	@echo "  NVIDIA_SRC=<path>  NVIDIA driver source for nv-p2p.h"
