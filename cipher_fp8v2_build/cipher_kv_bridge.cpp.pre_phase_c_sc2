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

py::object page_info(uint64_t devptr) {
	struct cipher_rt_kv_page_tag tag;
	if (cipher_rt_kv_page_info(devptr, &tag) != 0)
		return py::none();
	py::dict d;
	d["tenant_id"] = tag.tenant_id;
	d["seq_id"] = tag.seq_id;
	d["layer"] = (int)tag.layer;
	d["head_kv"] = (int)tag.head_kv;
	d["role"] = (int)tag.role;
	d["content_hash"] = tag.content_hash;
	return d;
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
}
