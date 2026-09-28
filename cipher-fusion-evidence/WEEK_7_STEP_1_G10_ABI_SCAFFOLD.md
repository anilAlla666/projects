# WEEK_7_STEP_1_G10_ABI_SCAFFOLD.md

**Date:** 2026-05-23
**Step:** v1.2.3 §7 W7-9 Step 1 — G10 ABI scaffolding (CIPHER_REGISTER_MODEL ioctl NR 27 + model registry + model_uuid field in `cipher_pid_stats`).
**Pre-tag:** `week-6-step-g1-g2-cap-bump` (`c4e2d6f`, kmod 0.5.0).
**Close tag:** **`week-7-step-1-g10-abi-scaffold`** (`a9d18aa`, kmod 0.5.5).
**Scope-lock parent:** `WEEK_7_9_SCOPE_LOCK.md` §3 (cipher-fusion-evidence `58c03e4`).

---

## 1. Pre-condition verification

| Anchor | Expected | Disk | Match |
|---|---|---|---|
| cipher-fusion-evidence HEAD | post-scope-lock | `58c03e4` | ✓ |
| cipher_kmod | tag `week-6-step-g1-g2-cap-bump` | `c4e2d6f` (kmod 0.5.0, srcversion `92F650B7FE869D5C0BA569B`) | ✓ |
| cipher_rt_phase4 | `ec0e005` (week-5-complete) | `ec0e005…` | ✓ |
| ioctl NR 26 highest | `CIPHER_DSM_PROPOSE` at `cipher_ioctl.h:558-559` | confirmed | ✓ |
| ioctl NR 27 free | additive per `[[cipher-abi-rule]]` | confirmed unused | ✓ |
| `cipher_pid_stats reserved[14] = 56 B` | post-W1 Cb.2 bump | confirmed `cipher_internal.h:310` | ✓ |

**Baselines preserved at `/tmp/g10_baseline/`**:

| Pre-file | md5 |
|---|---|
| `cipher_kmod.ko.pre` | `2f294edf1df8bc0996bd03b84c096f58` (W6 G1+G2 close) |
| `cipher_ioctl.h.pre` | `32325773c8c2881571712eb4fe5c016d` |
| `cipher_internal.h.pre` | `5714207cda5a403a16d084ed221759e9` |
| `cipher_main.c.pre` | `749b851285589e85509bf6da720791e9` |
| `cipher_proc.c.pre` | `cd845af5ad3b15453e67842bb4a67cdf` |
| `Kbuild.pre` | `28abfb22d1939ee19d115ab5cb4ce898` |
| `cipher_vllm_kv.py.pre` | `52fd291e4ad58a3cba5b58532e5853c1` |

---

## 2. The edits — kmod side (cipher_kmod, 6 modified + 1 new)

### 2.1 `cipher_ioctl.h` — NR 27 + payload struct + enum + sentinel (`+42` lines, after L559)

```c
enum cipher_model_arch {
    CIPHER_MODEL_ARCH_MISTRAL  = 1,
    CIPHER_MODEL_ARCH_QWEN     = 2,
    CIPHER_MODEL_ARCH_LLAMA    = 3,
    CIPHER_MODEL_ARCH_GPT_NEOX = 4,
    CIPHER_MODEL_ARCH_OTHER    = 255,
};

struct cipher_register_model {
    char     model_path[PATH_MAX];   /* in : canonical model dir path */
    __u8     hf_config_hash[32];     /* in : sha256 over hf_config canonical JSON */
    __u32    model_arch;             /* in : enum cipher_model_arch */
    __u8     model_uuid[16];         /* out: 128-bit UUID */
    __u32    flags;                  /* in/out: reserved, set to 0 */
    __u8     reserved[12];
};

#define CIPHER_MODEL_UUID_UNKNOWN_INIT { 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 }
#define CIPHER_REGISTER_MODEL \
    _IOWR(CIPHER_IOCTL_MAGIC, 27, struct cipher_register_model)
```

**sizeof(struct cipher_register_model) = 4164 B** (PATH_MAX 4096 + 32 + 4 + 16 + 4 + 12). Heap-allocated in the ioctl handler per the W6 G1+G2 `cipher_arena_query` kzalloc precedent (avoids the 1024 B kernel-stack budget).

### 2.2 `cipher_internal.h` — `model_uuid` field in `cipher_pid_stats` (`+18 / -1`)

```c
/* W7-9 Step 1 G10: per-tenant model identity. 16 B from reserved tail
 * (56 B -> 40 B remaining). Sentinel all-zero = MODEL_UNKNOWN.
 * Step 2 will consume 8 B (commit_seq) leaving 32 B. Step 3 HMAC
 * accumulator (64 B) WILL force struct grow -> ABI bump 0.5.5 -> 0.6.0. */
u8       model_uuid[16];
u32      reserved[10];             /* was reserved[14]; -16 B for model_uuid */
```

Plus prototype declarations after the Track-2 weight-arena block (around L424):

```c
int  cipher_model_registry_init(void);
void cipher_model_registry_exit(void);
int  cipher_model_register(const struct cipher_register_model *req,
                           u8 out_uuid[16]);
int  cipher_model_lookup_by_uuid(const u8 uuid[16],
                                 u8 hf_config_hash_out[32],
                                 u32 *model_arch_out);
long cipher_dev_register_model(unsigned long arg);
```

Userspace mirror `cipher_tenant_snapshot_user` size stays at 336 B (model_uuid is internal kmod state, not yet exposed in the snapshot). Step 2 will add `commit_seq` to the snapshot mirror; Step 3 will bump the mirror to 400 B for the G6 HMAC accumulator.

### 2.3 `cipher_model_registry.c` (new file, 220 LOC)

Hashtable keyed on FNV-32 of `hf_config_hash[0..3]` (256 buckets via `DEFINE_HASHTABLE(cipher_model_hash, 8)`). Spinlock-protected register + lookup. `get_random_bytes(16)` for uuid generation. Idempotent: same hash always returns the same uuid. Race-safe double-check pattern on insert (lookup outside lock → kzalloc outside lock → re-lookup under lock → use existing if a peer raced, otherwise insert and return).

Public API:
- `cipher_model_registry_init(void)` / `cipher_model_registry_exit(void)` — module lifecycle
- `cipher_model_register(req, out_uuid)` — idempotent register
- `cipher_model_lookup_by_uuid(uuid, hash_out, arch_out)` — reverse lookup (consumed by W10-12 G3+G4 / W13-14 G12)
- `cipher_dev_register_model(arg)` — ioctl handler; heap-allocates the 4 KiB payload

### 2.4 `cipher_dev.c:290` — dispatch case (`+2`)

```c
case CIPHER_REGISTER_MODEL:               /* W7-9 Step 1 G10 (nr 27) */
    return cipher_dev_register_model(arg);
```

Inserted after `case CIPHER_DSM_PROPOSE` (L289), before the `-ENOSYS` block at L291.

### 2.5 `cipher_main.c` — init/exit + MODULE_VERSION (`+4 / -1`)

- `cipher_model_registry_init()` added at L58 after `cipher_wa_init()`.
- `cipher_model_registry_exit()` added at L127 before `cipher_wa_exit()` (reverse order).
- `MODULE_VERSION("0.5.0") → MODULE_VERSION("0.5.5")` at L141 with W7-9 changelog comment.

### 2.6 `cipher_proc.c:155` — banner string (`+1 / -1`)

```c
- "cipher_kmod 0.5.0  uptime=%llu jiffies (%llu.%02llu s)\n",
+ "cipher_kmod 0.5.5  uptime=%llu jiffies (%llu.%02llu s)\n",
```

### 2.7 `Kbuild` — new object (`+1`)

```
cipher_kvdedup.o \
cipher_flops.o \
cipher_model_registry.o
```

### 2.8 Diff stat (kmod commit `a9d18aa`)

```
 Kbuild                  |   3 +-
 cipher_dev.c            |   2 +
 cipher_internal.h       |  18 +++-
 cipher_ioctl.h          |  42 +++++++++
 cipher_main.c           |   4 +-
 cipher_model_registry.c | 220 ++++++++++++++++++++++++++++++++++++++++++++++++
 cipher_proc.c           |   2 +-
 7 files changed, 287 insertions(+), 4 deletions(-)
```

**287 LOC total** (vs spec's ~200 LOC ballpark; overshoot is the docstring + race-check pattern in `cipher_model_registry.c`, both intentional).

---

## 3. Build + load verification (Part C)

### 3.1 Build

`make clean && make`:

- Errors: 0
- Warnings: 1 (compiler-vs-kernel env, matches baseline)
- `cipher_kmod.ko` produced
- New md5 **`2e36cd99276082eab228ded34121898b`** (was `2f294edf…` at W6 close)
- `modinfo`: `version=0.5.5`, `srcversion=C919E44DC2336F0F60A99A4` (was `92F650B7FE869D5C0BA569B`)

### 3.2 Load

```
$ sudo rmmod cipher_kmod
$ sudo insmod cipher_kmod.ko
[dmesg]
cipher_kmod: loading (Phase 4 — CP 5.4 8-SM-group arbitration; W4 Step 4 retired LP-8)
cipher_kmod: CP 5.4 arbitration ledger — 15 × 8-SM groups (120 SMs); legacy nr-9 4-SM allocator deactivated
cipher_kmod: Track 2 SC5 weight-arena registry — 100 slots, 5s liveness reaper
cipher_kmod: model_registry ready (W7-9 Step 1 G10) — 256-bucket hashtable    ← G10 visible
cipher_kmod: CP 3.3 FLOP telemetry ready (/proc/cipher/flops, ioctl nr 11/12)
cipher_kmod: /dev/cipher ready (major=511, ...)
cipher_kmod: /dev/cipher_kvdedup ready (major=510, ...)
...
cipher_kmod: loaded ok; nvidia_unlocked_ioctl @ 0xffffffffc05f6300 hooked

$ cat /proc/cipher/stats | head -1
cipher_kmod 0.5.5  uptime=...                                                  ← banner 0.5.5
```

No `KERN_WARNING` or `KERN_ERR`.

### 3.3 `test_register_model.c` — 5/5 PASS

Test source: `/tmp/g10_baseline/test_register_model.c` (200 LOC).

| Case | Description | Result |
|---|---|---|
| 1 | REGISTER_MODEL with Mistral hash → non-zero uuid | **PASS** — uuid `d495a9edcaf06dd697f16101c9cbf1f5` |
| 2 | Re-register same Mistral hash → idempotent same uuid | **PASS** |
| 3 | Distinct hash (Qwen) → distinct uuid `345fc73ec9b7fb8972f2c497ff30920a` | **PASS** |
| 4 | 5 concurrent threads × same Llama hash → 5/5 agree on uuid | **PASS** |
| 5 | 1000-iter mean round-trip = **0.50 µs** (budget 100 µs) | **PASS — 200× under budget** |

---

## 4. The edits — plugin side (`cipher_vllm_plugin`, 1 modified)

### 4.1 `cipher_vllm_kv.py` (`+150 LOC`; not git-tracked)

| Metric | Pre | Post |
|---|---|---|
| md5 | `52fd291e4ad58a3cba5b58532e5853c1` | **`0870311e376004ac83038c4d7e7ea882`** |
| lines | ~142 | **293** |

Additions:

- Module-level `_IOC` helper, `CipherRegisterModel` ctypes mirror (sizeof = 4164 B verified at import, matches kmod byte-for-byte), `_REGISTER_MODEL_IOCTL` constant.
- `_MODEL_TYPE_TO_ARCH` map: `"mistral" → 1`, `"qwen2"|"qwen" → 2`, `"llama" → 3`, `"gpt_neox"|"gpt-neox" → 4`, else → `255 (OTHER)`.
- `_canonical_hf_config_hash(hf_config)` — sha256 over a deterministic JSON of dimensional fields (`model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `vocab_size`, `max_position_embeddings`, `tie_word_embeddings`, `torch_dtype`). Stable identity for downstream model-keyed actuators.
- `_register_model_with_kmod(model_path, hf_config)` — opens `/dev/cipher`, fills `CipherRegisterModel`, issues `fcntl.ioctl(fd, CIPHER_REGISTER_MODEL, req, True)`, returns 16-byte uuid. **Backward-compat**: `-ENOSYS` → log warning, return `MODEL_UNKNOWN` sentinel (all-zero); downstream actuators treat as pass-through.
- `cipher_model_uuid()` — public accessor for downstream consumers.
- Wired into `_cipher_allocate_kv_cache_tensors` on first invocation per engine: pulls `self.model_config.hf_config` + `self.model_config.model` path, calls `_register_model_with_kmod`, caches uuid in module-level `_MODEL_UUID`. Defensive `try/except`: any failure leaves uuid at `MODEL_UNKNOWN`.
- New env gate `CIPHER_REGISTER_MODEL=0` for explicit opt-out.

Plugin is not version-controlled; the new md5 `0870311e…` is the canonical post-Step-1 snapshot. Track via memory anchor + this doc.

---

## 5. Regression gates (Part E) — all PASS

### 5.1 E.1 W6 G1+G2 cap regression

- `test_cap65 N=128`: **128/128** succeed (byte-identical to `week-6-step-g1-g2-cap-bump` close).
- `test_arena17 N=100`: **100/100** succeed.

### 5.2 E.2 CP 5.4 isolation

`cp_5_4/step1_3/cp54_isolation_test`: **15/15 PASS**, byte-identical to v1.2.2 baseline.

### 5.3 E.3 libcipher_rt.so loader smoke

`libcipher_rt.so` md5 **`259ac994aead2da8289fc84d6116fbe9`** (unchanged — Step 1 doesn't touch libcipher_rt).
Symbols resolve clean: `cipher_rt_tenant_cached` ✓ `cipher_rt_matmul_dispatch_init` ✓ `cipher_rt_attn_dispatch_init` ✓.

### 5.4 E.4 vLLM Mistral-7B B=1 decode smoke

Command (matched W6 Option-1 baseline parameters):

```
bench_llm.py --model mistralai/Mistral-7B-v0.1 --batch-sizes 1 \
  --iterations 3 --warmup-iters 2 --output-tokens 256 \
  --input-length-distribution fixed:1024 --workload agentic --cipher \
  --gpu-mem-util 0.50
```

Results (run_id `g10_smoke_mistral7b_256tok`):

| Iter | agg_tps | decode_MFU% | mean_W | peakHBM_MiB | smClk_p50 |
|---|---|---|---|---|---|
| 1 | 163.6 | 0.249 | 432 | 41642 | 1830 |
| 2 | 163.5 | 0.249 | 431 | 41642 | 1830 |
| 3 | 163.4 | 0.249 | 434 | 41642 | 1830 |

- **Mean TPS: 163.5 tok/s** vs W6 baseline 163.3 → **delta +0.12%**, well within ±3% gate.
- decode_MFU 0.249% byte-identical to W6.
- `REGISTER_MODEL OK: arch=1 uuid=5c30f3f70c698687a10352d3836497ec path='mistralai/Mistral-7B-v0.1'` observed in stderr — fires once at engine init.

### 5.5 E.5 Heterogeneous-model smoke (Mistral + Qwen distinct uuids)

Direct Python invocation of the plugin's `_register_model_with_kmod` with synthesized `types.SimpleNamespace` hf_configs:

| Model | model_arch | model_uuid |
|---|---|---|
| Mistral-7B (mistralai/Mistral-7B-v0.1) | 1 | `5c30f3f70c698687a10352d3836497ec` |
| Qwen2-7B (Qwen/Qwen2-7B) | 2 | `6b0e7b67e064982ac9154ce8055880eb` |

- **Distinct**: Mistral uuid ≠ Qwen uuid.
- **Idempotent across processes**: re-registering Mistral in a separate Python invocation returned the same uuid (the kmod's hashtable persisted between vLLM E.4 run and the E.5 Python smoke).

---

## 6. Final state + anchors

| Field | Value |
|---|---|
| `cipher_kmod` HEAD | **`a9d18aa`** (tag `week-7-step-1-g10-abi-scaffold`) |
| `cipher_kmod.ko` md5 (pre) | `2f294edf1df8bc0996bd03b84c096f58` (W6 close) |
| `cipher_kmod.ko` md5 (post) | **`2e36cd99276082eab228ded34121898b`** |
| `srcversion` | **`C919E44DC2336F0F60A99A4`** |
| `MODULE_VERSION` | **`0.5.5`** |
| `/proc/cipher/stats` banner | **`cipher_kmod 0.5.5`** |
| `cipher_vllm_kv.py` md5 (pre) | `52fd291e4ad58a3cba5b58532e5853c1` |
| `cipher_vllm_kv.py` md5 (post) | **`0870311e376004ac83038c4d7e7ea882`** |
| `libcipher_rt.so` md5 | `259ac994aead2da8289fc84d6116fbe9` (unchanged) |
| `cipher_rt_phase4` HEAD | `ec0e005…` (unchanged) |

---

## 7. v1.2.3 §7 W7-9 progression

- ✓ **Step 1 G10 ABI scaffolding** (this commit) — close tag `week-7-step-1-g10-abi-scaffold` (cipher_kmod `a9d18aa`).
- Step 2 COMMIT primitive — pre-tag `week-7-step-1-g10-abi-scaffold` (this); close tag `week-7-step-2-commit-primitive`.
- Step 3 G6 kmod-resident AUDIT chain — pre-tag `week-7-step-2-commit-primitive`; close tag `week-8-step-3-g6-audit-chain`. ABI bump 0.5.5 → 0.6.0 (struct size grow).
- Step 4 21 overlay-ops `_report()` port — pre-tag `week-8-step-3-g6-audit-chain`; close tag `week-8-step-4-overlay-ops-port`.
- Step 5 N=128 contention soak — pre-tag `week-8-step-4-overlay-ops-port`; close tag `week-9-step-5-n128-soak` = `week-9-complete`.

Next prompt: **Step 2 COMMIT primitive core** per `WEEK_7_9_SCOPE_LOCK.md` §4 (`cipher_rt_phase4` tag `week-7-step-2-commit-primitive`; ~3 eng-days; ~200 LOC userspace + ~50 LOC kmod).
