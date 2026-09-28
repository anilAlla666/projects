/* SPDX-License-Identifier: GPL-2.0-or-later
 * D.10 LT-ROUTE — cublasLt descriptor decode (see cipher_rt_cublaslt_layout.c). */
#ifndef CIPHER_RT_CUBLASLT_LAYOUT_H
#define CIPHER_RT_CUBLASLT_LAYOUT_H
#include "cipher_rt_matmul_dispatch.h"

int cipher_rt_cublaslt_layout_init(void);

/* Decode opaque cublasLt descriptors into a gemmEx-shaped matmul_call.
 * Returns 0 on a fully-consistent decode (call populated), -1 otherwise
 * (caller MUST pass through to the real variant — never route on -1). */
int cipher_rt_cublaslt_decode(void *computeDesc,
                              const void *A, void *Adesc,
                              const void *B, void *Bdesc,
                              const void *C, void *Cdesc,
                              void *D, void *Ddesc,
                              const void *alpha, const void *beta,
                              void *stream,
                              struct cipher_rt_matmul_call *call);
#endif
