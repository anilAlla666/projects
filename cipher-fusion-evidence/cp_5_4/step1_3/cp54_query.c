/* CP 5.4 Step 1.3 Phase D — /dev/cipher CP 5.4 ledger QUERY tool (THROWAWAY).
 * Opens /dev/cipher, issues CIPHER_CP54_QUERY, prints the ledger snapshot.
 * The caller's own pid is a fresh non-allocating pid, so my_grp_mask is
 * normally 0; n_partitions / pool_grp_count / free_grp_count are global. */
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include "cipher_ioctl.h"

int main(void)
{
	struct cipher_cp54_query q;
	int fd = open("/dev/cipher", O_RDWR);

	if (fd < 0) { perror("open /dev/cipher"); return 2; }
	if (ioctl(fd, CIPHER_CP54_QUERY, &q) != 0) { perror("CP54_QUERY"); return 2; }
	close(fd);
	printf("CP54_QUERY n_partitions=%u pool_grp_count=%u free_grp_count=%u "
	       "my_grp_mask=0x%04x my_qos_class=0x%x\n",
	       q.n_partitions, q.pool_grp_count, q.free_grp_count,
	       q.my_grp_mask, q.my_qos_class);
	return 0;
}
