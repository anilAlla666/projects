#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0xe3ec2f2b, "alloc_chrdev_region" },
	{ 0x8d522714, "__rcu_read_lock" },
	{ 0xe1688e24, "proc_create" },
	{ 0xeea0e0d, "class_destroy" },
	{ 0x96848186, "scnprintf" },
	{ 0x37a0cba, "kfree" },
	{ 0x7ab1f0cb, "pcpu_hot" },
	{ 0x7369f212, "seq_lseek" },
	{ 0xba8fbd64, "_raw_spin_lock" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x122c3a7e, "_printk" },
	{ 0xf0fdf6cb, "__stack_chk_fail" },
	{ 0x87a21cb3, "__ubsan_handle_out_of_bounds" },
	{ 0x6a7b86fa, "cdev_add" },
	{ 0xb7c0f443, "sort" },
	{ 0x6091797f, "synchronize_rcu" },
	{ 0x2469810f, "__rcu_read_unlock" },
	{ 0x4ed0e44, "device_create" },
	{ 0xa4bf0f83, "class_create" },
	{ 0x4c03a563, "random_kmalloc_seed" },
	{ 0x8b797b14, "seq_putc" },
	{ 0x87b77d87, "unregister_kretprobe" },
	{ 0xa4eda9c3, "proc_mkdir" },
	{ 0x827acc57, "__get_task_comm" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0xac3c422b, "proc_remove" },
	{ 0x15ba50a6, "jiffies" },
	{ 0xc1d9b323, "seq_read" },
	{ 0x44c10a52, "kvfree_call_rcu" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0x6445ca15, "device_destroy" },
	{ 0xc00e2b80, "seq_printf" },
	{ 0x3f66a26e, "register_kprobe" },
	{ 0x984866c0, "register_kretprobe" },
	{ 0x76b8104b, "seq_puts" },
	{ 0x8e66928c, "single_release" },
	{ 0xbb10e61d, "unregister_kprobe" },
	{ 0xbf55f104, "kmalloc_trace" },
	{ 0x60a13e90, "rcu_barrier" },
	{ 0x46b0be46, "single_open" },
	{ 0xb5b54b34, "_raw_spin_unlock" },
	{ 0x81daace6, "cdev_init" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0x1004e946, "kmalloc_caches" },
	{ 0x67d01ca4, "cdev_del" },
	{ 0x73776b79, "module_layout" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "252730F830B7807CEF3C7C8");
