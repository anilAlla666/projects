/* T4.3.2 — quick test of CIPHER_SET_CLOCK_MHZ (nr 10) ioctl.
 *
 * Usage:
 *   sudo /tmp/cipher_set_clock_test <mhz>
 *   sudo /tmp/cipher_set_clock_test 0    # reset/unlock
 *
 * Requires CAP_SYS_ADMIN, so run with sudo. The deployment plumbing
 * (libcipher_rt → ioctl → kmod → nvidia-smi) does NOT need sudo
 * because the kmod's call_usermodehelper inherits root privilege.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/ioctl.h>
#include <linux/types.h>
#include <errno.h>
#include <unistd.h>

#define CIPHER_IOCTL_MAGIC  'C'
#define CIPHER_SET_CLOCK_MHZ  _IOW(CIPHER_IOCTL_MAGIC, 10, __u32)

int main(int argc, char **argv)
{
	__u32 mhz;
	int fd, rc;

	if (argc != 2) {
		fprintf(stderr, "usage: %s <mhz>\n", argv[0]);
		return 1;
	}
	mhz = (__u32)atoi(argv[1]);

	fd = open("/dev/cipher", O_RDWR);
	if (fd < 0) {
		perror("open /dev/cipher");
		return 2;
	}

	rc = ioctl(fd, CIPHER_SET_CLOCK_MHZ, &mhz);
	if (rc < 0) {
		fprintf(stderr, "ioctl rc=%d errno=%d (%s)\n", rc, errno, strerror(errno));
		close(fd);
		return 3;
	}

	printf("CIPHER_SET_CLOCK_MHZ(%u) ok\n", mhz);
	close(fd);
	return 0;
}
