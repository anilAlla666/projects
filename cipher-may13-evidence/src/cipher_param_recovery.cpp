// CIPHER kernel parameter recovery — Phase 1.
//
// Builds a hostFun → kernel-info registry by intercepting cudart's two
// internal symbols:
//   __cudaRegisterFatBinary(void* fatCubin) → void**        // 'handle'
//   __cudaRegisterFunction(void** handle, hostFun, deviceFun, deviceName, …)
//
// PyTorch (and any other CUDA-using process) calls these at module-load
// time for every compiled kernel. The host_fun is the same address later
// passed as `func` to cudaLaunchKernel.
//
// Fatbin / cubin layouts are not officially documented, but the relevant
// magic numbers and section formats are stable enough across CUDA 11/12 to
// hand-parse for what we need:
//   * Fatbin wrapper: { magic=0x466243B1, version, fatbin_blob_ptr, ... }
//   * Fatbin header:  { magic=0xBA55ED50, version, headerSize, fatSize }
//   * Fatbin sections: a list of (header, payload) where one payload kind
//     holds an ELF cubin
//   * Cubin ELF: standard ELF64; per-kernel attribute records live in
//     `.nv.info.<funcname>` sections, format records keyed by EIATTR_*
//   * EIATTR_KPARAM_INFO (0x17) records: 4×u32 holding (kid, ordinal,
//     offset, size_align)
//
// For the first cut we record the function table eagerly and lazy-parse
// the cubin sections only when a kernel is actually queried — that keeps
// startup cheap and doesn't touch fatbins for kernels we never look up.

#include "cipher_param_recovery.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <elf.h>

namespace {

constexpr uint32_t FATBIN_WRAPPER_MAGIC = 0x466243B1;
constexpr uint32_t FATBIN_HEADER_MAGIC  = 0xBA55ED50;

struct FatbinWrapper {
    uint32_t magic;
    uint32_t version;
    const void* data;            // pointer to fatbin header
    const void* filename;
};

struct FatbinHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t headerSize;
    uint64_t fatSize;            // size of payload region after this header
};

// EIATTR codes we care about
constexpr uint8_t EIATTR_KPARAM_INFO = 0x17;
constexpr uint8_t EIFMT_NVAL         = 0x01;   // no value (we skip)
constexpr uint8_t EIFMT_BVAL         = 0x02;   // 1-byte value
constexpr uint8_t EIFMT_HVAL         = 0x03;   // 2-byte value
constexpr uint8_t EIFMT_SVAL         = 0x04;   // size-prefixed value

struct KernelEntry {
    CipherParamEntry e;
    std::mutex parse_mu;
};

struct State {
    std::mutex                                   table_mu;
    std::unordered_map<const void*, KernelEntry*> by_host;
    std::unordered_map<std::string, KernelEntry*> by_name;
    std::unordered_map<const void*, KernelEntry*> by_func;     // CUfunction → entry
    std::unordered_map<void*, const void*>       fatbin_blob;
    std::atomic<int>                             enabled{0};
    std::atomic<uint64_t>                        fatbins_seen{0};
    std::atomic<uint64_t>                        functions_seen{0};
    std::atomic<uint64_t>                        parsed_ok{0};
    std::atomic<uint64_t>                        parse_failed{0};
};

State& S() {
    static State s;
    return s;
}

// ── Fatbin parsing ─────────────────────────────────────────────────────────
//
// The fatbin header is followed by zero or more sections. Each section has
// its own subheader; the on-disk layout has been observed (across CUDA 11
// and 12) as:
//
//   uint16_t kind;       // 1 = PTX text, 2 = ELF cubin
//   uint16_t unknown0;
//   uint32_t headerSize;
//   uint64_t size;       // payload size in bytes
//   ...padding fields...
//   payload[size]
//
// We only look at ELF cubins (kind==2). The headerSize is needed to skip
// past variable-length headers in newer fatbins.

bool parse_cubin_for_kernel(const uint8_t* elf, size_t elf_size,
                              const char* func_name,
                              KernelEntry* out)
{
    if (elf_size < sizeof(Elf64_Ehdr)) return false;
    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(elf);
    if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) return false;
    if (ehdr->e_ident[EI_CLASS] != ELFCLASS64) return false;

    if (ehdr->e_shoff == 0 || ehdr->e_shnum == 0) return false;
    const Elf64_Shdr* shdr = reinterpret_cast<const Elf64_Shdr*>(elf + ehdr->e_shoff);
    if (ehdr->e_shstrndx >= ehdr->e_shnum) return false;
    const Elf64_Shdr& strtab_shdr = shdr[ehdr->e_shstrndx];
    if (strtab_shdr.sh_offset + strtab_shdr.sh_size > elf_size) return false;
    const char* strtab = reinterpret_cast<const char*>(elf + strtab_shdr.sh_offset);

    // Build expected `.nv.info.<funcname>` name.
    char target[512];
    snprintf(target, sizeof(target), ".nv.info.%s", func_name);

    for (size_t i = 0; i < ehdr->e_shnum; ++i) {
        if (shdr[i].sh_name >= strtab_shdr.sh_size) continue;
        const char* sec_name = strtab + shdr[i].sh_name;
        if (strcmp(sec_name, target) != 0) continue;
        if (shdr[i].sh_offset + shdr[i].sh_size > elf_size) continue;

        const uint8_t* p   = elf + shdr[i].sh_offset;
        const uint8_t* end = p + shdr[i].sh_size;

        int n_params = 0;
        while (p + 4 <= end) {
            uint8_t  fmt   = p[0];
            uint8_t  attr  = p[1];
            uint16_t vsize = (uint16_t)p[2] | ((uint16_t)p[3] << 8);
            p += 4;
            size_t adv = 0;
            switch (fmt) {
                case EIFMT_NVAL: adv = 0; break;
                case EIFMT_BVAL: adv = 1; break;
                case EIFMT_HVAL: adv = 2; break;
                case EIFMT_SVAL: adv = vsize; break;
                default:         adv = vsize; break;
            }
            if (p + adv > end) break;

            if (attr == EIATTR_KPARAM_INFO && fmt == EIFMT_SVAL && adv >= 16) {
                // 4×u32 record. word 1 has the parameter ordinal in bits 16..23
                // (older format) or directly (newer format); word 2 = byte
                // offset; word 3 low 16 = size, low byte = log2(align).
                uint32_t w1, w2, w3;
                memcpy(&w1, p + 4,  4);
                memcpy(&w2, p + 8,  4);
                memcpy(&w3, p + 12, 4);
                int      ordinal = (int)((w2 >> 16) & 0xFFFF);
                uint16_t offset  = (uint16_t)(w2 & 0xFFFF);
                // Some encodings put the offset in word2 directly and ordinal
                // in word1>>16. Try the canonical first; if the ordinal looks
                // bogus, fall through to alternate.
                if (ordinal >= 16) {
                    ordinal = (int)((w1 >> 16) & 0xFFFF);
                    offset  = (uint16_t)(w2 & 0xFFFF);
                }
                uint16_t size = (uint16_t)(w3 & 0xFFFF);
                if (ordinal < 16 && n_params < 16) {
                    if (ordinal + 1 > n_params) n_params = ordinal + 1;
                    out->e.params[ordinal].offset = offset;
                    out->e.params[ordinal].size   = size;
                }
            }
            p += adv;
        }
        out->e.param_count = n_params;
        out->e.parsed_ok   = 1;
        return n_params > 0;
    }
    return false;
}

// Walk a cubin and add ALL its kernels (sections matching `.nv.info.<name>`)
// into the by_name table. Returns the count of kernels found.
int parse_cubin_all_kernels(const uint8_t* payload, size_t payload_size,
                              State& s)
{
    // NVCC cubin payloads sometimes have a 1-byte prefix before the ELF
    // magic. Scan the first 4 bytes to find \x7fELF.
    size_t skip = 0;
    while (skip <= 4 && skip + SELFMAG <= payload_size) {
        if (memcmp(payload + skip, ELFMAG, SELFMAG) == 0) break;
        skip++;
    }
    if (skip > 4) return 0;
    const uint8_t* elf = payload + skip;
    size_t elf_size    = payload_size - skip;

    if (elf_size < sizeof(Elf64_Ehdr)) return 0;
    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(elf);
    if (ehdr->e_ident[EI_CLASS] != ELFCLASS64) return 0;
    if (ehdr->e_shoff == 0 || ehdr->e_shnum == 0) return 0;
    // CUDA cubins use a non-standard e_shentsize (NVCC extends section hdrs).
    // We use the value reported in the file rather than sizeof(Elf64_Shdr).
    size_t shentsize = ehdr->e_shentsize;
    if (shentsize < sizeof(Elf64_Shdr) || shentsize > 256) return 0;
    if (ehdr->e_shoff + (size_t)ehdr->e_shnum * shentsize > elf_size) return 0;

    auto get_shdr = [&](size_t i) -> const Elf64_Shdr* {
        return reinterpret_cast<const Elf64_Shdr*>(
            elf + ehdr->e_shoff + i * shentsize);
    };
    if (ehdr->e_shstrndx >= ehdr->e_shnum) return 0;
    const Elf64_Shdr* strtab_shdr = get_shdr(ehdr->e_shstrndx);
    if (strtab_shdr->sh_offset + strtab_shdr->sh_size > elf_size) return 0;
    const char* strtab = reinterpret_cast<const char*>(elf + strtab_shdr->sh_offset);

    int added = 0;
    for (size_t i = 0; i < ehdr->e_shnum; ++i) {
        const Elf64_Shdr* sh = get_shdr(i);
        if (sh->sh_name >= strtab_shdr->sh_size) continue;
        const char* sec_name = strtab + sh->sh_name;
        if (strncmp(sec_name, ".nv.info.", 9) != 0) continue;
        const char* fname = sec_name + 9;
        if (sh->sh_offset + sh->sh_size > elf_size) continue;

        std::string key = fname;
        {
            std::lock_guard<std::mutex> lk(s.table_mu);
            if (s.by_name.count(key)) continue;     // already have it
        }

        // Tentative entry — will be filled in below.
        KernelEntry* k = new KernelEntry{};
        // Persist the name in a heap copy.
        char* name_copy = strdup(fname);
        k->e.device_name = name_copy;
        k->e.device_fun  = name_copy;

        const uint8_t* p   = elf + sh->sh_offset;
        const uint8_t* end = p + sh->sh_size;

        int n_params = 0;
        int safety   = 0;
        while (p + 4 <= end && safety < 4096) {
            safety++;
            uint8_t  fmt   = p[0];
            uint8_t  attr  = p[1];
            uint16_t vsize = (uint16_t)p[2] | ((uint16_t)p[3] << 8);
            p += 4;
            size_t adv = 0;
            switch (fmt) {
                case EIFMT_NVAL: adv = 0; break;
                case EIFMT_BVAL: adv = 1; break;
                case EIFMT_HVAL: adv = 2; break;
                case EIFMT_SVAL: adv = vsize; break;
                default:         adv = vsize; break;
            }
            if (p + adv > end) break;

            if (attr == EIATTR_KPARAM_INFO && fmt == EIFMT_SVAL && adv >= 16) {
                uint32_t w1, w2, w3;
                memcpy(&w1, p + 4,  4);
                memcpy(&w2, p + 8,  4);
                memcpy(&w3, p + 12, 4);
                int      ordinal = (int)((w2 >> 16) & 0xFFFF);
                uint16_t offset  = (uint16_t)(w2 & 0xFFFF);
                if (ordinal >= 16) {
                    ordinal = (int)((w1 >> 16) & 0xFFFF);
                    offset  = (uint16_t)(w2 & 0xFFFF);
                }
                uint16_t size = (uint16_t)(w3 & 0xFFFF);
                if (ordinal < 16) {
                    if (ordinal + 1 > n_params) n_params = ordinal + 1;
                    k->e.params[ordinal].offset = offset;
                    k->e.params[ordinal].size   = size;
                }
            }
            p += adv;
        }
        k->e.param_count = n_params;
        k->e.parsed_ok   = (n_params > 0) ? 1 : 0;

        {
            std::lock_guard<std::mutex> lk(s.table_mu);
            // Re-check (race-safe insert).
            if (s.by_name.count(key)) {
                free(name_copy);
                delete k;
                continue;
            }
            s.by_name[key] = k;
        }
        added++;
        s.functions_seen.fetch_add(1, std::memory_order_relaxed);
        if (n_params > 0) s.parsed_ok.fetch_add(1, std::memory_order_relaxed);
        else              s.parse_failed.fetch_add(1, std::memory_order_relaxed);
    }
    return added;
}

// Walk a fatbin blob and ingest every cubin into the global table.
int parse_fatbin_all_kernels(const uint8_t* base, size_t base_size, State& s)
{
    if (base_size < sizeof(FatbinHeader)) return 0;
    const FatbinHeader* hdr = reinterpret_cast<const FatbinHeader*>(base);
    if (hdr->magic != FATBIN_HEADER_MAGIC) return 0;
    if (hdr->headerSize < sizeof(FatbinHeader)) return 0;
    if (hdr->headerSize + hdr->fatSize > base_size) return 0;

    const uint8_t* p   = base + hdr->headerSize;
    const uint8_t* end = base + hdr->headerSize + hdr->fatSize;
    int added = 0;

    while (p + 8 < end) {
        uint16_t kind = ((uint16_t)p[1] << 8) | p[0];
        uint32_t hsize = 0;
        uint64_t psize = 0;
        memcpy(&hsize, p + 4,  4);
        memcpy(&psize, p + 8,  8);
        if (hsize < 16 || hsize > 1024) break;
        if (p + hsize + psize > end)    break;

        if (kind == 2) {
            added += parse_cubin_all_kernels(p + hsize, (size_t)psize, s);
        }
        p += hsize + psize;
    }
    s.fatbins_seen.fetch_add(1, std::memory_order_relaxed);
    return added;
}

// Walk a fatbin blob, find an ELF cubin, parse it for `func_name`.
bool parse_fatbin_for_kernel(const void* fatbin_blob,
                               const char* func_name,
                               KernelEntry* out)
{
    if (!fatbin_blob) return false;
    const FatbinHeader* hdr = reinterpret_cast<const FatbinHeader*>(fatbin_blob);
    if (hdr->magic != FATBIN_HEADER_MAGIC) return false;
    if (hdr->headerSize < sizeof(FatbinHeader)) return false;

    const uint8_t* base = reinterpret_cast<const uint8_t*>(fatbin_blob);
    const uint8_t* p   = base + hdr->headerSize;
    const uint8_t* end = base + hdr->headerSize + hdr->fatSize;

    while (p + 8 < end) {
        uint16_t kind     = ((uint16_t)p[1] << 8) | p[0];
        // Section header size is at offset 4 (u32). Section payload size at
        // offset 8 (u64). On newer fatbins the layout has more fields but
        // these two are stable.
        uint32_t hsize    = 0;
        uint64_t psize    = 0;
        memcpy(&hsize, p + 4,  4);
        memcpy(&psize, p + 8,  8);
        if (hsize < 16 || hsize > 1024) return false;
        if (p + hsize + psize > end)    return false;

        if (kind == 2 /* ELF cubin */) {
            if (parse_cubin_for_kernel(p + hsize, (size_t)psize,
                                         func_name, out))
                return true;
        }
        p += hsize + psize;
    }
    return false;
}

void ensure_parsed_locked(KernelEntry* k) {
    if (k->e.parsed_ok != 0 || k->e.param_count > 0) return;
    auto& s = S();
    const void* blob = nullptr;
    {
        std::lock_guard<std::mutex> lk(s.table_mu);
        auto it = s.fatbin_blob.find(k->e.fat_handle);
        if (it != s.fatbin_blob.end()) blob = it->second;
    }
    if (!blob) {
        s.parse_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const char* fname = k->e.device_fun ? k->e.device_fun : k->e.device_name;
    if (!fname) {
        s.parse_failed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (parse_fatbin_for_kernel(blob, fname, k)) {
        s.parsed_ok.fetch_add(1, std::memory_order_relaxed);
    } else {
        s.parse_failed.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace

extern "C" int cipher_param_scan_dso(const char* path) {
    if (!path) return 0;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return 0;
    }
    void* mem = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mem == MAP_FAILED) return 0;

    const uint8_t* data = reinterpret_cast<const uint8_t*>(mem);
    size_t size = st.st_size;

    int total_added = 0;
    State& s = S();

    // Scan for FATBIN_HEADER_MAGIC.
    for (size_t i = 0; i + sizeof(FatbinHeader) < size; ) {
        const FatbinHeader* h = reinterpret_cast<const FatbinHeader*>(data + i);
        if (h->magic == FATBIN_HEADER_MAGIC
            && h->headerSize >= sizeof(FatbinHeader)
            && h->headerSize <= 256
            && h->fatSize > 0
            && h->fatSize < (1ull << 31)
            && i + h->headerSize + h->fatSize <= size) {
            total_added += parse_fatbin_all_kernels(data + i, size - i, s);
            i += h->headerSize + h->fatSize;
        } else {
            i += 4;
        }
    }
    munmap(mem, size);
    return total_added;
}

extern "C" int cipher_param_scan_pytorch(void) {
    static const char* paths[] = {
        "/usr/lib/python3/dist-packages/torch/lib/libtorch_cuda.so",
        "/usr/lib/python3/dist-packages/torch/lib/libtorch_cuda_linalg.so",
        "/usr/lib/x86_64-linux-gnu/libcublas.so.12",
        "/usr/lib/x86_64-linux-gnu/libcublasLt.so.12",
    };
    int total = 0;
    for (const char* p : paths) {
        int n = cipher_param_scan_dso(p);
        if (n > 0)
            fprintf(stderr, "[CIPHER PARAM] scan %s: +%d kernels\n", p, n);
        total += n;
    }
    return total;
}

extern "C" const CipherParamEntry*
cipher_param_lookup_by_name(const char* kernel_name)
{
    if (!kernel_name) return nullptr;
    auto& s = S();
    std::lock_guard<std::mutex> lk(s.table_mu);
    auto it = s.by_name.find(kernel_name);
    if (it == s.by_name.end()) return nullptr;
    return &it->second->e;
}

// Driver-level cubin ingest. The blob is whatever cuModuleLoadData(Ex)
// receives — could be a raw cubin ELF, a fatbin, or PTX. Detect by magic.
extern "C" int cipher_param_ingest_image(const void* image)
{
    if (!image) return 0;
    // Default OFF — we now use cuKernelGetParamInfo via the driver instead
    // of parsing cubins ourselves (set CIPHER_PARAM_INGEST=1 to re-enable
    // the parsing path, which is significantly slower at startup).
    static int gate = -1;
    if (gate < 0) {
        const char* g = getenv("CIPHER_PARAM_INGEST");
        gate = (g && (g[0] == '1' || strcmp(g, "on") == 0)) ? 1 : 0;
    }
    if (!gate) return 0;

    State& s = S();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(image);

    // Raw ELF cubin (most common path from static cudart): \x7fELF
    if (memcmp(p, ELFMAG, SELFMAG) == 0) {
        // Without an explicit size, derive an upper bound from the ELF
        // header itself: sh_offset + n*sh_entsize + max(sh_size). The
        // parse function only reads inside its own stated bounds, so we
        // can pass a generous limit safely.
        const Elf64_Ehdr* eh = reinterpret_cast<const Elf64_Ehdr*>(p);
        size_t max_sec_end = 0;
        if (eh->e_shoff && eh->e_shnum && eh->e_shentsize >= sizeof(Elf64_Shdr)
            && eh->e_shentsize <= 256)
        {
            for (size_t i = 0; i < eh->e_shnum; ++i) {
                const Elf64_Shdr* sh = reinterpret_cast<const Elf64_Shdr*>(
                    p + eh->e_shoff + i * eh->e_shentsize);
                size_t end = (size_t)sh->sh_offset + (size_t)sh->sh_size;
                if (end > max_sec_end) max_sec_end = end;
            }
        }
        size_t guess = max_sec_end ? max_sec_end + 4096 : (16u << 20);
        return parse_cubin_all_kernels(p, guess, s);
    }

    // Fatbin (rare for cuModuleLoadData but possible — driver accepts).
    if (((const FatbinHeader*)p)->magic == FATBIN_HEADER_MAGIC) {
        const FatbinHeader* h = (const FatbinHeader*)p;
        size_t total = (size_t)h->headerSize + (size_t)h->fatSize;
        return parse_fatbin_all_kernels(p, total, s);
    }

    return 0;
}

extern "C" void
cipher_param_register_cufunc(const void* cufunc, const char* name)
{
    if (!cufunc || !name) return;
    auto& s = S();
    std::lock_guard<std::mutex> lk(s.table_mu);
    auto it = s.by_name.find(name);
    if (it == s.by_name.end()) return;     // we don't have this kernel
    s.by_func[cufunc] = it->second;
}

extern "C" const CipherParamEntry*
cipher_param_lookup_cufunc(const void* cufunc)
{
    if (!cufunc) return nullptr;
    auto& s = S();
    std::lock_guard<std::mutex> lk(s.table_mu);
    auto it = s.by_func.find(cufunc);
    if (it == s.by_func.end()) return nullptr;
    return &it->second->e;
}

extern "C" int cipher_param_recovery_init(void) {
    // Default ON — recovery is cheap when nothing queries it.
    S().enabled.store(1, std::memory_order_relaxed);
    return 1;
}

extern "C" int cipher_param_recovery_enabled(void) {
    return S().enabled.load(std::memory_order_relaxed);
}

extern "C" void cipher_param_register_fatbin(void* handle, const void* fatbin_blob) {
    if (!handle) return;
    auto& s = S();
    // Newer cudart wraps the fatbin pointer in a small wrapper struct; if so,
    // unwrap it.
    const void* blob = fatbin_blob;
    if (blob) {
        const FatbinWrapper* wrap = reinterpret_cast<const FatbinWrapper*>(blob);
        if (wrap->magic == FATBIN_WRAPPER_MAGIC) blob = wrap->data;
    }
    {
        std::lock_guard<std::mutex> lk(s.table_mu);
        s.fatbin_blob[handle] = blob;
    }
    s.fatbins_seen.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void cipher_param_register_function(void* fat_handle,
                                                  const void* host_fun,
                                                  const char* device_fun,
                                                  const char* device_name)
{
    if (!host_fun) return;
    auto& s = S();
    {
        std::lock_guard<std::mutex> lk(s.table_mu);
        if (s.by_host.count(host_fun)) return;
        auto* k = new KernelEntry{};
        k->e.host_fun     = host_fun;
        k->e.device_fun   = device_fun;
        k->e.device_name  = device_name;
        k->e.fat_handle   = fat_handle;
        s.by_host[host_fun] = k;
    }
    s.functions_seen.fetch_add(1, std::memory_order_relaxed);
}

extern "C" const CipherParamEntry* cipher_param_lookup(const void* host_fun) {
    if (!host_fun) return nullptr;
    auto& s = S();
    KernelEntry* k = nullptr;
    {
        std::lock_guard<std::mutex> lk(s.table_mu);
        auto it = s.by_host.find(host_fun);
        if (it == s.by_host.end()) return nullptr;
        k = it->second;
    }
    {
        std::lock_guard<std::mutex> lk2(k->parse_mu);
        ensure_parsed_locked(k);
    }
    return &k->e;
}

extern "C" void cipher_param_stats(CipherParamStats* out) {
    if (!out) return;
    auto& s = S();
    out->fatbins_seen   = s.fatbins_seen.load(std::memory_order_relaxed);
    out->functions_seen = s.functions_seen.load(std::memory_order_relaxed);
    out->parsed_ok      = s.parsed_ok.load(std::memory_order_relaxed);
    out->parse_failed   = s.parse_failed.load(std::memory_order_relaxed);
}

extern "C" void cipher_param_table_dump(int to_file) {
    auto& s = S();
    FILE* f = stderr;
    if (to_file) f = fopen("/tmp/cipher_param_table.txt", "w");
    if (!f) f = stderr;

    fprintf(f, "=== CIPHER param-recovery table ===\n");
    CipherParamStats st{};
    cipher_param_stats(&st);
    fprintf(f, "fatbins_seen=%llu  functions_seen=%llu  parsed_ok=%llu  parse_failed=%llu\n",
            (unsigned long long)st.fatbins_seen,
            (unsigned long long)st.functions_seen,
            (unsigned long long)st.parsed_ok,
            (unsigned long long)st.parse_failed);
    fprintf(f, "by_func resolved=%zu\n", S().by_func.size());

    std::lock_guard<std::mutex> lk(s.table_mu);
    int idx = 0;
    fprintf(f, "\n--- by_host (%zu entries) ---\n", s.by_host.size());
    for (auto& kv : s.by_host) {
        KernelEntry* k = kv.second;
        {
            std::lock_guard<std::mutex> lk2(k->parse_mu);
            ensure_parsed_locked(k);
        }
        const char* nm = k->e.device_name ? k->e.device_name :
                          (k->e.device_fun ? k->e.device_fun : "<?>");
        fprintf(f, "[%4d] host=%p name=%.140s nparams=%d parsed_ok=%d\n",
                idx, k->e.host_fun, nm, k->e.param_count, k->e.parsed_ok);
        idx++;
    }
    int idx2 = 0;
    fprintf(f, "\n--- by_name (%zu entries — fatbin scan) ---\n", s.by_name.size());
    for (auto& kv : s.by_name) {
        KernelEntry* k = kv.second;
        const char* nm = k->e.device_name ? k->e.device_name : "<?>";
        fprintf(f, "[%5d] nparams=%d  name=%.150s\n",
                idx2, k->e.param_count, nm);
        for (int i = 0; i < k->e.param_count; ++i) {
            fprintf(f, "         param[%d]: offset=%u size=%u\n",
                    i, (unsigned)k->e.params[i].offset,
                       (unsigned)k->e.params[i].size);
        }
        idx2++;
        if (idx2 >= 200 && f != stderr) {
            // Truncate file dump after 200 entries.
            fprintf(f, "... (truncated; %zu more)\n", s.by_name.size() - idx2);
            break;
        }
    }
    if (f != stderr) {
        fclose(f);
        fprintf(stderr, "[CIPHER PARAM] table dumped to /tmp/cipher_param_table.txt (%d entries)\n", idx);
    }
}

__attribute__((constructor(113)))
static void cipher_param_recovery_autoinit() { cipher_param_recovery_init(); }
