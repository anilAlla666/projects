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

KSYMTAB_FUNC(cipher_get_current_tenant_snapshot, "_gpl", "");
KSYMTAB_FUNC(cipher_get_tenant_snapshot_by_pid, "_gpl", "");
KSYMTAB_FUNC(cipher_get_tenant_snapshot_by_id, "_gpl", "");
KSYMTAB_FUNC(cipher_enumerate_tenants, "_gpl", "");
KSYMTAB_FUNC(cipher_partition_request, "_gpl", "");
KSYMTAB_FUNC(cipher_partition_request_v2, "_gpl", "");
KSYMTAB_FUNC(cipher_partition_release, "_gpl", "");
KSYMTAB_FUNC(cipher_partition_release_slots_only, "_gpl", "");

SYMBOL_CRC(cipher_get_current_tenant_snapshot, 0xe793e37d, "_gpl");
SYMBOL_CRC(cipher_get_tenant_snapshot_by_pid, 0x4112574f, "_gpl");
SYMBOL_CRC(cipher_get_tenant_snapshot_by_id, 0xd15b4d4b, "_gpl");
SYMBOL_CRC(cipher_enumerate_tenants, 0x28181765, "_gpl");
SYMBOL_CRC(cipher_partition_request, 0x9151b72a, "_gpl");
SYMBOL_CRC(cipher_partition_request_v2, 0x3238399f, "_gpl");
SYMBOL_CRC(cipher_partition_release, 0xc1df361f, "_gpl");
SYMBOL_CRC(cipher_partition_release_slots_only, 0x3012c43b, "_gpl");

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0xa78af5f3, "ioread32" },
	{ 0xe3ec2f2b, "alloc_chrdev_region" },
	{ 0x13c49cc2, "_copy_from_user" },
	{ 0x8d522714, "__rcu_read_lock" },
	{ 0xe1688e24, "proc_create" },
	{ 0x94a18028, "pci_iomap" },
	{ 0x656e4a6e, "snprintf" },
	{ 0xcba997cd, "pci_dev_put" },
	{ 0x2b4bb393, "pci_get_device" },
	{ 0x74c134b9, "__sw_hweight32" },
	{ 0xeea0e0d, "class_destroy" },
	{ 0x96848186, "scnprintf" },
	{ 0x53569707, "this_cpu_off" },
	{ 0xa08ca28d, "fget" },
	{ 0xe54d9269, "fd_install" },
	{ 0x37a0cba, "kfree" },
	{ 0x7ab1f0cb, "pcpu_hot" },
	{ 0x7369f212, "seq_lseek" },
	{ 0xb3f7646e, "kthread_should_stop" },
	{ 0xba8fbd64, "_raw_spin_lock" },
	{ 0xcc5005fe, "msleep_interruptible" },
	{ 0xcbd4898c, "fortify_panic" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0xecc30390, "wake_up_process" },
	{ 0x7f24de73, "jiffies_to_usecs" },
	{ 0x122c3a7e, "_printk" },
	{ 0xf0fdf6cb, "__stack_chk_fail" },
	{ 0xa916b694, "strnlen" },
	{ 0xc6cbbc89, "capable" },
	{ 0x87a21cb3, "__ubsan_handle_out_of_bounds" },
	{ 0x599fb41c, "kvmalloc_node" },
	{ 0x6a7b86fa, "cdev_add" },
	{ 0x9114b616, "__xa_alloc" },
	{ 0xcaa82808, "fput" },
	{ 0xb7c0f443, "sort" },
	{ 0x6091797f, "synchronize_rcu" },
	{ 0x2469810f, "__rcu_read_unlock" },
	{ 0xa7eedcc4, "call_usermodehelper" },
	{ 0x4ed0e44, "device_create" },
	{ 0xa85a3e6d, "xa_load" },
	{ 0xa4bf0f83, "class_create" },
	{ 0x4c03a563, "random_kmalloc_seed" },
	{ 0x4dfa8d4b, "mutex_lock" },
	{ 0x5a921311, "strncmp" },
	{ 0x8b797b14, "seq_putc" },
	{ 0x87b77d87, "unregister_kretprobe" },
	{ 0x689f3974, "kthread_stop" },
	{ 0x89940875, "mutex_lock_interruptible" },
	{ 0x71cf81bb, "pci_iounmap" },
	{ 0xa4eda9c3, "proc_mkdir" },
	{ 0x92ec510d, "jiffies64_to_msecs" },
	{ 0x827acc57, "__get_task_comm" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x6b10bee1, "_copy_to_user" },
	{ 0xac3c422b, "proc_remove" },
	{ 0x15ba50a6, "jiffies" },
	{ 0x3817aecb, "kthread_create_on_node" },
	{ 0x745a981, "xa_erase" },
	{ 0xc1d9b323, "seq_read" },
	{ 0xdd64e639, "strscpy" },
	{ 0xa648e561, "__ubsan_handle_shift_out_of_bounds" },
	{ 0x44c10a52, "kvfree_call_rcu" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0x3213f038, "mutex_unlock" },
	{ 0xa843805a, "get_unused_fd_flags" },
	{ 0xd9491c14, "xa_destroy" },
	{ 0x8fa25c24, "xa_find" },
	{ 0x6445ca15, "device_destroy" },
	{ 0xc00e2b80, "seq_printf" },
	{ 0x3f66a26e, "register_kprobe" },
	{ 0x984866c0, "register_kretprobe" },
	{ 0x76b8104b, "seq_puts" },
	{ 0x8e66928c, "single_release" },
	{ 0xbb10e61d, "unregister_kprobe" },
	{ 0x73eb4e65, "from_kuid_munged" },
	{ 0xdf36914b, "xa_find_after" },
	{ 0xbf55f104, "kmalloc_trace" },
	{ 0x60a13e90, "rcu_barrier" },
	{ 0x54b1fac6, "__ubsan_handle_load_invalid_value" },
	{ 0xa7e3af6c, "param_ops_int" },
	{ 0x7aa1756e, "kvfree" },
	{ 0x46b0be46, "single_open" },
	{ 0xb5b54b34, "_raw_spin_unlock" },
	{ 0x81daace6, "cdev_init" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0xe2c17b5d, "__SCT__might_resched" },
	{ 0x1004e946, "kmalloc_caches" },
	{ 0x67d01ca4, "cdev_del" },
	{ 0x73776b79, "module_layout" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "E427CAFA4E94D548233DC7A");
