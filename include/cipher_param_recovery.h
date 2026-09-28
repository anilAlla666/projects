// CIPHER kernel parameter recovery — Phase 1.
//
// Intercepts __cudaRegisterFatBinary / __cudaRegisterFunction at PyTorch
// startup so we can build a table:
//
//     hostFun  →  { deviceName, fatbin_data, parameter_count,
//                    parameter_offsets[], parameter_sizes[] }
//
// `hostFun` is the same value PyTorch later passes to cudaLaunchKernel as
// the first argument, so any launch can be looked up against this table to
// recover the original kernel name and per-parameter layout. Phase 4 uses
// this to identify which arg index of (e.g.) the RoPE kernel is the output
// pointer, so we can quantize it.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CipherParamInfo {
    uint16_t offset;     // byte offset in the kernel param block
    uint16_t size;       // size in bytes
} CipherParamInfo;

// Pulled at parse time. param_count == 0 if parsing hasn't been attempted
// yet; parsed_ok==0 if parsing was attempted and failed.
typedef struct CipherKernelEntry {
    const void*       host_fun;          // PyTorch host stub address
    const char*       device_fun;        // mangled device function name (raw ptr from cudart)
    const char*       device_name;       // demangled-ish name (the deviceName arg)
    void*             fat_handle;        // opaque fatbin handle from __cudaRegisterFatBinary
    int               param_count;       // populated by lazy parse
    int               parsed_ok;
    CipherParamInfo   params[16];        // up to 16 params (typical kernels have ≤8)
} CipherKernelEntry;

// Probe APIs.
int cipher_param_recovery_init(void);
int cipher_param_recovery_enabled(void);

// Intercept hooks — called from libcipher_hook.so's __cuda* shims.
void  cipher_param_register_fatbin(void* handle, const void* fatbin_blob);
void  cipher_param_register_function(void* fat_handle,
                                       const void* host_fun,
                                       const char* device_fun,
                                       const char* device_name);

// Lookup: returns NULL if no entry. Threadsafe.
const CipherKernelEntry* cipher_param_lookup(const void* host_fun);

// Lookup by kernel name (mangled or demangled). Used when cudart_static is
// in play and we can't get a host_fun → name mapping via __cudaRegister*.
// We instead scan loaded DSOs for fatbin magic and build the table directly
// from cubin sections.
const CipherKernelEntry* cipher_param_lookup_by_name(const char* kernel_name);

// Scan a DSO file for embedded fatbins and parse all cubins. Idempotent;
// repeat calls re-use the cached table. Returns the number of NEW kernels
// added by this call.
int cipher_param_scan_dso(const char* path);

// Convenience: scan a curated list of PyTorch / CUDA DSOs likely to contain
// fatbins (libtorch_cuda.so, libcublas.so, libcublasLt.so). Returns total
// kernels added across all scans.
int cipher_param_scan_pytorch(void);

// Driver-level cubin ingest: called from the cuModuleLoadData(Ex) shim
// with the DECOMPRESSED cubin blob the driver received from cudart_static.
// Detects whether image is a raw cubin ELF, a fatbin, or PTX; parses
// `.nv.info.<funcname>` sections; populates by_name table. Returns the
// number of new kernels added.
int  cipher_param_ingest_image(const void* image);

// Map a CUfunction handle to a kernel name. Called from the
// cuModuleGetFunction shim. Subsequent cuLaunchKernel(f, ...) calls can
// be resolved to (name, param layout) via cipher_param_lookup_cufunc.
void cipher_param_register_cufunc(const void* cufunc, const char* name);

// Lookup by CUfunction. Returns NULL if no entry.
const CipherKernelEntry* cipher_param_lookup_cufunc(const void* cufunc);

// Dump the full table to stderr (or a file at /tmp/cipher_param_table.txt
// when arg `to_file != 0`). Used by the Phase 1 probe.
void  cipher_param_table_dump(int to_file);

// Stats.
typedef struct CipherParamStats {
    uint64_t fatbins_seen;
    uint64_t functions_seen;
    uint64_t parsed_ok;
    uint64_t parse_failed;
} CipherParamStats;

void  cipher_param_stats(CipherParamStats* out);

#ifdef __cplusplus
}
#endif
