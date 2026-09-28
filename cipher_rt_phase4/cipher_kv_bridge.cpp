// SPDX-License-Identifier: GPL-2.0-or-later
//
// cipher_kv_bridge.cpp - Phase 4.6.2 S2b: from_blob bridge.
//
// A small pybind11 / libtorch extension module. It is the seam between
// the C VMM page allocator (cipher_rt_kv_alloc) and Python: it creates
// a CIPHER-owned VMM slab and wraps it as a torch.Tensor via
// at::from_blob, so transformers' StaticLayer can hold KV cache tensors
// whose physical memory CIPHER owns and tags.
//
// torch::from_blob is C++-only — there is no clean public Python API to
// wrap an external CUDA pointer as a tensor, hence this extension.
//
// Build: see build_kv_bridge.sh. Importable as `cipher_kv_bridge`.

#include <torch/extension.h>
#include <cuda.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

extern "C" {
#include "cipher_rt_kv_alloc.h"
}

namespace {

at::ScalarType dtype_from_string(const std::string &s) {
	if (s == "float16" || s == "half")     return at::kHalf;
	if (s == "bfloat16")                   return at::kBFloat16;
	if (s == "float32" || s == "float")    return at::kFloat;
	/* CP 5.1: vLLM v1 allocates the raw KV buffer as a flat int8 tensor
	 * (gpu_model_runner._allocate_kv_cache_tensors) and reshapes it later
	 * with .view(dtype). The bridge must hand back an int8 tensor of
	 * exactly `size` bytes for that path. */
	if (s == "int8")                       return at::kChar;
	throw std::runtime_error("cipher_kv_bridge: unsupported dtype '" + s + "'");
}

bool bridge_init(int64_t va_pool_bytes) {
	return cipher_rt_kv_alloc_init((size_t)va_pool_bytes) == 0;
}

// Create a CIPHER VMM slab sized for `shape` of `dtype`, fully mapped
// (StaticLayer pre-allocates the whole [B,H,max_len,D] tensor — the
// attention kernel reads the full span, so every page must be backed),
// zero-initialized by the allocator, and return it wrapped as a tensor.
// The tensor's deleter releases the slab when Python GCs it.
torch::Tensor vmm_zeros(std::vector<int64_t> shape, int64_t itemsize,
                        std::string dtype, uint32_t tenant, uint32_t seq,
                        uint32_t layer, uint32_t role) {
	int64_t numel = 1;
	for (auto d : shape) numel *= d;
	size_t nbytes = (size_t)numel * (size_t)itemsize;

	struct cipher_rt_kv_page_tag tag;
	tag.tenant_id = tenant;
	tag.seq_id = seq;
	tag.layer = (uint16_t)layer;
	tag.head_kv = 0xFFFF;          /* whole-tensor slab: spans all KV heads */
	tag.role = (uint8_t)role;
	tag._pad[0] = tag._pad[1] = tag._pad[2] = 0;
	tag.content_hash = 0;          /* T4.6.3 fills this */

	unsigned long long devptr = 0;
	cipher_rt_kv_slab_t *slab =
		cipher_rt_kv_slab_create(&tag, nbytes, nbytes, &devptr);
	if (!slab || !devptr)
		throw std::runtime_error("cipher_kv_bridge: slab_create failed "
		                         "(VA pool exhausted?)");

	auto opts = torch::TensorOptions()
	                .dtype(dtype_from_string(dtype))
	                .device(torch::kCUDA, 0);
	// Deleter releases the CIPHER slab when the tensor is destroyed.
	return torch::from_blob(
		(void *)devptr, shape,
		[slab](void *) { cipher_rt_kv_slab_free(slab); },
		opts);
}

// --- Phase C SC2: producer-side weight arena -------------------------
// A weight arena is one VMM allocation (cipher_rt_weight_arena_*). Weight
// tensors are bump-allocated sub-ranges wrapped via from_blob with a
// no-op deleter — the arena, not the tensor, owns the physical memory.
// Hold the WeightArena object alive as long as any tensor from it is in
// use (SC2 has no kmod-owned lifetime yet — that is SC5).
struct WeightArena {
	unsigned long long base = 0;
	size_t             size = 0;   // usable bytes (C side rounds up to page)
	size_t             cursor = 0;
	uint32_t           tenant = 0;
	bool               live = false;
	~WeightArena() {
		if (live) cipher_rt_weight_arena_free(base);
	}
};

std::shared_ptr<WeightArena> weight_arena_create(int64_t nbytes,
                                                 uint32_t tenant) {
	if (nbytes <= 0)
		throw std::runtime_error("cipher_kv_bridge: weight_arena_create "
		                         "needs nbytes > 0");
	auto a = std::make_shared<WeightArena>();
	unsigned long long base = 0;
	if (cipher_rt_weight_arena_create(tenant, (size_t)nbytes, &base) != 0)
		throw std::runtime_error("cipher_kv_bridge: weight_arena_create "
		                         "failed");
	a->base = base;
	a->size = (size_t)nbytes;
	a->cursor = 0;
	a->tenant = tenant;
	a->live = true;
	return a;
}

// Bump-allocate a weight tensor inside the arena (512-byte aligned).
torch::Tensor weight_arena_alloc(std::shared_ptr<WeightArena> a,
                                 std::vector<int64_t> shape,
                                 int64_t itemsize, std::string dtype) {
	if (!a || !a->live)
		throw std::runtime_error("cipher_kv_bridge: weight arena not live");
	int64_t numel = 1;
	for (auto d : shape) numel *= d;
	size_t nbytes = (size_t)numel * (size_t)itemsize;
	size_t off = (a->cursor + 511) & ~((size_t)511);
	if (off + nbytes > a->size)
		throw std::runtime_error("cipher_kv_bridge: weight arena exhausted");
	void *ptr = (void *)(a->base + off);
	a->cursor = off + nbytes;
	auto opts = torch::TensorOptions()
	                .dtype(dtype_from_string(dtype))
	                .device(torch::kCUDA, 0);
	// no-op deleter: the WeightArena owns the physical memory.
	return torch::from_blob(ptr, shape, [](void *) {}, opts);
}

// Export the arena as a POSIX fd — the producer-side shareable handle.
// The caller owns the returned fd.
int weight_arena_export(std::shared_ptr<WeightArena> a) {
	if (!a || !a->live)
		throw std::runtime_error("cipher_kv_bridge: weight arena not live");
	int fd = -1;
	if (cipher_rt_weight_arena_export(a->base, &fd) != 0)
		throw std::runtime_error("cipher_kv_bridge: weight_arena_export "
		                         "failed");
	return fd;
}

// Track 2 SC3 — CONSUMER side. Import a peer's exported weight arena (fd from
// weight_arena_export, passed cross-process). want_base = the producer's base
// (for a same-VA mapping; pass 0 to skip). Returns a live WeightArena whose
// physical memory is the shared copy — its destructor unmaps + releases.
std::shared_ptr<WeightArena> weight_arena_import(int fd,
                                                 unsigned long long want_base,
                                                 int64_t nbytes) {
	if (nbytes <= 0)
		throw std::runtime_error("cipher_kv_bridge: weight_arena_import "
		                         "needs nbytes > 0");
	auto a = std::make_shared<WeightArena>();
	unsigned long long base = 0;
	if (cipher_rt_weight_arena_import(fd, want_base, (size_t)nbytes,
	                                  &base) != 0)
		throw std::runtime_error("cipher_kv_bridge: weight_arena_import "
		                         "failed");
	a->base = base;
	a->size = (size_t)nbytes;
	a->cursor = 0;
	a->tenant = 0;
	a->live = true;
	return a;
}

// Track 2 SC3 — a tensor ALIASING the arena at an explicit offset (consumer
// side; no bump-allocation, no copy). Used to rebind a meta-loaded model's
// parameters onto an imported arena per the layout manifest.
torch::Tensor weight_arena_view(std::shared_ptr<WeightArena> a,
                                std::vector<int64_t> shape,
                                int64_t itemsize, std::string dtype,
                                int64_t offset) {
	if (!a || !a->live)
		throw std::runtime_error("cipher_kv_bridge: weight arena not live");
	int64_t numel = 1;
	for (auto d : shape) numel *= d;
	size_t nbytes = (size_t)numel * (size_t)itemsize;
	if (offset < 0 || (size_t)offset + nbytes > a->size)
		throw std::runtime_error("cipher_kv_bridge: weight_arena_view "
		                         "range exceeds arena");
	void *ptr = (void *)(a->base + (size_t)offset);
	auto opts = torch::TensorOptions()
	                .dtype(dtype_from_string(dtype))
	                .device(torch::kCUDA, 0);
	// no-op deleter: the WeightArena owns the physical memory.
	return torch::from_blob(ptr, shape, [](void *) {}, opts);
}

py::object page_info(uint64_t devptr) {
	// KV slab page?
	struct cipher_rt_kv_page_tag tag;
	if (cipher_rt_kv_page_info(devptr, &tag) == 0) {
		py::dict d;
		d["kind"] = "kv";
		d["tenant_id"] = tag.tenant_id;
		d["seq_id"] = tag.seq_id;
		d["layer"] = (int)tag.layer;
		d["head_kv"] = (int)tag.head_kv;
		d["role"] = (int)tag.role;
		d["content_hash"] = tag.content_hash;
		return d;
	}
	// weight-arena page?
	uint32_t wt = 0;
	unsigned long long wh = 0;
	if (cipher_rt_weight_arena_info(devptr, &wt, &wh) == 0) {
		py::dict d;
		d["kind"] = "weight";
		d["tenant_id"] = wt;
		d["vmm_handle"] = wh;
		return d;
	}
	return py::none();   // not CIPHER-managed
}

py::dict get_stats() {
	struct cipher_rt_kv_alloc_stats st;
	cipher_rt_kv_alloc_get_stats(&st);
	py::dict d;
	d["slabs_created"] = st.slabs_created;
	d["slabs_freed"] = st.slabs_freed;
	d["pages_mapped"] = st.pages_mapped;
	d["pages_unmapped"] = st.pages_unmapped;
	d["pages_resident"] = st.pages_resident;
	d["pages_resident_peak"] = st.pages_resident_peak;
	d["bytes_va_reserved"] = st.bytes_va_reserved;
	d["page_size"] = st.page_size;
	d["map_ms_total"] = st.map_ms_total;
	return d;
}

}  // namespace

PYBIND11_MODULE(cipher_kv_bridge, m) {
	m.doc() = "CIPHER T4.6.2 KV VMM allocator bridge";
	m.def("init", &bridge_init, "Initialize the VMM allocator",
	      py::arg("va_pool_bytes"));
	m.def("vmm_zeros", &vmm_zeros,
	      "Allocate a zero-initialized CIPHER-VMM-backed tensor",
	      py::arg("shape"), py::arg("itemsize"), py::arg("dtype"),
	      py::arg("tenant"), py::arg("seq"), py::arg("layer"),
	      py::arg("role"));
	m.def("page_info", &page_info, "Reverse-lookup a device pointer's tag",
	      py::arg("devptr"));
	m.def("get_stats", &get_stats, "Allocator telemetry snapshot");

	// Phase C SC2 — producer-side weight arena.
	py::class_<WeightArena, std::shared_ptr<WeightArena>>(m, "WeightArena")
	    .def("alloc", &weight_arena_alloc,
	         "Bump-allocate a weight tensor inside the arena",
	         py::arg("shape"), py::arg("itemsize"), py::arg("dtype"))
	    .def("export_fd", &weight_arena_export,
	         "Export the arena as a POSIX fd (caller owns it)")
	    .def("view", &weight_arena_view,
	         "Track 2 SC3 — alias the arena at an explicit byte offset",
	         py::arg("shape"), py::arg("itemsize"), py::arg("dtype"),
	         py::arg("offset"))
	    .def_readonly("base", &WeightArena::base)
	    .def_readonly("size", &WeightArena::size)
	    .def_readonly("tenant", &WeightArena::tenant)
	    .def_property_readonly("cursor",
	         [](const WeightArena &a) { return a.cursor; });
	m.def("weight_arena_create", &weight_arena_create,
	      "Create a producer-side VMM weight arena",
	      py::arg("nbytes"), py::arg("tenant"));
	// Phase C SC3 — consumer-side import.
	m.def("weight_arena_import", &weight_arena_import,
	      "Track 2 SC3 — import a peer's exported weight arena",
	      py::arg("fd"), py::arg("want_base"), py::arg("nbytes"));
}
