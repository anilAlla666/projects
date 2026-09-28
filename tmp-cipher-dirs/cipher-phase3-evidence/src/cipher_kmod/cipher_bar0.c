// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_bar0.c -- direct BAR0 register reads from H100.
 *
 * !!! READ-ONLY !!!
 * WRITES TO BAR0 WILL CORRUPT NVIDIA.KO STATE.
 * PHASE 3 IS OBSERVATION ONLY.
 *
 * No iowrite32. No __raw_writel. No memcpy_toio.
 * All reads go through cipher_bar0_rd32() which takes
 * const void __iomem * -- writes are impossible at the
 * type level.
 *
 * pci_iomap obtains the mapping; pci_iounmap releases it
 * at module unload. nvidia.ko already holds the device but
 * pci_iomap only ioremap()s the BAR -- it does not call
 * pci_request_region, so coexisting mappings are legal.
 *
 * Safe offsets WHITELIST (only these may be read):
 *   BAR0_PMC_BOOT_0        0x00000000   (cached at init)
 *   BAR0_PMC_BOOT_1        0x00000004   (cached at init)
 *   BAR0_PMC_INTR_0        0x00000100   (live)
 *   BAR0_PMC_INTR_1        0x00000104   (live)
 *   BAR0_PBUS_INTR_STATUS  0x00001100   (live)
 *
 * Do NOT read offsets in PFIFO (0x800000+), PRAMIN, or any
 * region nvidia.ko actively writes to.
 *
 * Layer A scope (minimal): bind, map, log raw PMC_BOOT_0/1.
 * Field decode (architecture, implementation, revision) is
 * deferred to Phase 6. The headline proof points are
 * (1) pci_iomap succeeds against an nvidia.ko-owned device,
 * (2) ioread32 returns real silicon (not all-ones), and
 * (3) module unload leaves no taint and no resource leak.
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/seq_file.h>

#include "cipher_internal.h"

#define BAR0_PMC_BOOT_0        0x00000000
#define BAR0_PMC_BOOT_1        0x00000004
#define BAR0_PMC_INTR_0        0x00000100
#define BAR0_PMC_INTR_1        0x00000104
#define BAR0_PBUS_INTR_STATUS  0x00001100

#define H100_BAR0_SIZE_BYTES   (16UL * 1024UL * 1024UL)

static struct pci_dev *cipher_pci_dev;
static void __iomem   *cipher_bar0_base;
static u32             cipher_bar0_pmc_boot_0;     /* cached at init */
static u32             cipher_bar0_pmc_boot_1;     /* cached at init */
static unsigned long   cipher_bar0_addr;           /* BAR0 phys addr */

static inline u32 cipher_bar0_rd32(const void __iomem *base, u32 offset)
{
	return ioread32(base + offset);
}

int cipher_bar0_init(void)
{
	struct pci_dev *pdev, *next;
	resource_size_t bar0_start, bar0_len;

	pdev = pci_get_device(PCI_VENDOR_ID_NVIDIA, PCI_ANY_ID, NULL);
	if (!pdev) {
		pr_info("cipher_bar0: no NVIDIA PCI device found; BAR0 layer disabled\n");
		return -ENODEV;
	}

	if ((pdev->class >> 16) != 0x03) {
		pr_warn("cipher_bar0: first NVIDIA device %04x:%04x not display-class (class=0x%06x); BAR0 layer disabled\n",
			pdev->vendor, pdev->device, pdev->class);
		pci_dev_put(pdev);
		return -ENODEV;
	}

	next = pci_get_device(PCI_VENDOR_ID_NVIDIA, PCI_ANY_ID, pdev);
	if (next) {
		pr_warn("cipher_bar0: multiple NVIDIA devices found; using first (%s). Multi-GPU is Phase 6.\n",
			pci_name(pdev));
		pci_dev_put(next);
	}

	bar0_start = pci_resource_start(pdev, 0);
	bar0_len   = pci_resource_len(pdev, 0);

	if (bar0_len == 0) {
		pr_err("cipher_bar0: BAR0 length is 0 on %04x:%04x; refusing to map\n",
		       pdev->vendor, pdev->device);
		pci_dev_put(pdev);
		return -EIO;
	}
	if (bar0_len != H100_BAR0_SIZE_BYTES) {
		pr_warn("cipher_bar0: BAR0 size %llu bytes (expected %lu MB on H100); proceeding anyway\n",
			(unsigned long long)bar0_len,
			H100_BAR0_SIZE_BYTES >> 20);
	}

	cipher_bar0_base = pci_iomap(pdev, 0, H100_BAR0_SIZE_BYTES);
	if (!cipher_bar0_base) {
		pr_err("cipher_bar0: pci_iomap(BAR0, %lu MB) failed on %s\n",
		       H100_BAR0_SIZE_BYTES >> 20, pci_name(pdev));
		pci_dev_put(pdev);
		return -EIO;
	}

	cipher_pci_dev          = pdev;
	cipher_bar0_addr        = (unsigned long)bar0_start;
	cipher_bar0_pmc_boot_0  = cipher_bar0_rd32(cipher_bar0_base, BAR0_PMC_BOOT_0);
	cipher_bar0_pmc_boot_1  = cipher_bar0_rd32(cipher_bar0_base, BAR0_PMC_BOOT_1);

	pr_info("cipher_bar0: bound to %s [%04x:%04x] BAR0=0x%lx size=%llu MB\n",
		pci_name(pdev), pdev->vendor, pdev->device,
		cipher_bar0_addr, (unsigned long long)(bar0_len >> 20));
	pr_info("cipher_bar0: PMC_BOOT_0=0x%08x (raw; decode deferred to Phase 6)\n",
		cipher_bar0_pmc_boot_0);
	pr_info("cipher_bar0: PMC_BOOT_1=0x%08x (raw; decode deferred to Phase 6)\n",
		cipher_bar0_pmc_boot_1);

	return 0;
}

void cipher_bar0_exit(void)
{
	if (cipher_bar0_base) {
		pci_iounmap(cipher_pci_dev, cipher_bar0_base);
		cipher_bar0_base = NULL;
	}
	if (cipher_pci_dev) {
		pci_dev_put(cipher_pci_dev);
		cipher_pci_dev = NULL;
	}
}

int cipher_bar0_proc_show(struct seq_file *m, void *v)
{
	u32 intr0, intr1, pbus_intr;

	if (!cipher_bar0_base) {
		seq_puts(m, "cipher_bar0: layer disabled (no NVIDIA GPU bound)\n");
		return 0;
	}

	seq_printf(m, "cipher_bar0 v0.3.1  bdf=%s\n", pci_name(cipher_pci_dev));
	seq_printf(m, "bar0_addr=0x%lx  size=%lu MB\n",
		   cipher_bar0_addr, H100_BAR0_SIZE_BYTES >> 20);
	seq_puts(m, "[REGISTERS -- read-only, no writes ever]\n");
	seq_printf(m, "PMC_BOOT_0 = 0x%08x (cached at init, raw)\n",
		   cipher_bar0_pmc_boot_0);
	seq_printf(m, "PMC_BOOT_1 = 0x%08x (cached at init, raw)\n",
		   cipher_bar0_pmc_boot_1);
	seq_puts(m, "  /* field decode (arch, impl, major_rev, minor_rev) deferred to Phase 6 */\n");

	intr0     = cipher_bar0_rd32(cipher_bar0_base, BAR0_PMC_INTR_0);
	intr1     = cipher_bar0_rd32(cipher_bar0_base, BAR0_PMC_INTR_1);
	pbus_intr = cipher_bar0_rd32(cipher_bar0_base, BAR0_PBUS_INTR_STATUS);

	seq_printf(m, "PMC_INTR(0) = 0x%08x  (live read on cat)\n", intr0);
	seq_printf(m, "PMC_INTR(1) = 0x%08x  (live read on cat)\n", intr1);
	seq_printf(m, "PBUS_INTR_STATUS = 0x%08x  (live read at offset 0x1100)\n",
		   pbus_intr);

	return 0;
}
