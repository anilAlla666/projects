/* W.7 test stub: provides cipher_nccl_record_decide so the tuner plugin's
 * dlsym(RTLD_DEFAULT) resolves -> exercises the ACTIVE decision path (item 4).
 * Returns a CIPHER_ALGO bucketed by message size to cover all mappings.
 * CIPHER_ALGO: AUTO=0 RING=1 TREE=2 NVLS=3 LL128=4 */
#include <stdint.h>
int cipher_nccl_record_decide(uint64_t bytes, uint32_t num_ranks) {
    (void)num_ranks;
    if (bytes < 65536)        return 4;  /* LL128 */
    if (bytes < 1048576)      return 2;  /* TREE  */
    if (bytes < 16777216)     return 1;  /* RING  */
    return 3;                            /* NVLS  */
}
