/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_marlin_kernel_src.h -- extern decl for the embedded Marlin
 * kernel CUDA source. The string is fed to NVRTC at libcipher_rt init,
 * compiled to a single cubin holding all instantiated entry points.
 *
 * Original kernel: IST-DASLab Marlin (Apache-2.0).
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

extern const char cipher_rt_marlin_kernel_src_str[];

#ifdef __cplusplus
}
#endif
