# cipher-platform v2.0 operator runbook

For the neocloud ops team operating CIPHER on GPU compute nodes. This runbook is shipped with the cipher-platform .deb at `/usr/lib/cipher/runbook.md`. End-users renting GPU compute do NOT need to read this; their `docker run --gpus all` invocations transparently pick up the substrate via nvidia-container-toolkit CDI.

## 1. Post-install verification

After `apt install cipher-platform`, verify the install in this order:

```sh
sudo cipher-platform status
sudo cipher-platform verify
```

Expected `status` output (v2.0 fresh install on a node with NVIDIA driver + nvidia-container-toolkit >= 1.17):

```
cipher-platform status
  dpkg version:      2.0
  substrate present: True (md5 1d91e7dabba6b3ace520d78fdb085fc3)
  kmod loaded:       True (version 0.6.5, /dev/cipher True)
  CDI patch:         applied=True marker=True
  plugin entries:    ['cipher_vllm_kv', 'cipher_vllm_kvdedup']
  systemd watcher:   active
```

Expected `verify` output: `cipher-platform verify: PASS` and exit 0.

End-user smoke test from the operator's perspective: launch a vanilla vLLM container and confirm the substrate fires:

```sh
docker run --rm --gpus all vllm/vllm-openai:0.21.0 \
  python3 -c "import ctypes, os; ctypes.CDLL('/usr/lib/cipher/libcipher_rt.so', mode=os.RTLD_LAZY); print('substrate visible in container')"
```

The CDI patch bind-mounts `/usr/lib/cipher/libcipher_rt.so` into the container; the import should succeed without the end-user setting any flags.

## 2. Post-driver-update refresh procedure

Major NVIDIA driver upgrades (e.g., `apt upgrade nvidia-driver-*`) cause nvidia-container-toolkit to regenerate `/var/run/cdi/nvidia.yaml`, which wipes the CIPHER patch. Recovery options:

**Option A (default, automatic):** the `cipher-platform-watch.path` systemd unit auto-detects the regeneration via inotify `PathChanged` and re-applies the CIPHER patch. Verify the watcher is running:

```sh
systemctl is-active cipher-platform-watch.path  # should be 'active'
sudo cipher-platform status                     # CDI patch should be applied=True
```

**Option B (manual fallback, if watcher disabled or failed):**

```sh
sudo cipher-platform refresh
sudo cipher-platform verify
```

If you intentionally disabled the watcher (`systemctl disable --now cipher-platform-watch.path`), make `cipher-platform refresh` a step in your post-driver-upgrade ops playbook.

## 3. Common debugging scenarios

### 3.1 CIPHER substrate not firing in end-user vLLM container

Symptoms: end-user reports no inference speedup; substrate visibility test from §1 fails.

Diagnose:

```sh
sudo cipher-platform verify --json
nvidia-ctk cdi list                  # should show nvidia.com/gpu=all
grep -A2 CIPHER /var/run/cdi/nvidia.yaml  # should show CIPHER mounts + env
dpkg -l nvidia-container-toolkit     # confirm >= 1.17
```

Common causes:
- nvidia-container-toolkit < 1.17 (no CDI mode). Fix: upgrade toolkit.
- CDI patch reverted by driver upgrade (see §2).
- End-user container uses non-standard Python layout (not `/usr/lib/python3/dist-packages`). Mitigation: if customer container is conda-based, they may need to add `/usr/lib/python3/dist-packages` to their container's PYTHONPATH; document the limitation in your end-user terms.
- End-user container set its own `LD_LIBRARY_PATH` that omits `/usr/lib/cipher`. CIPHER CDI sets `LD_LIBRARY_PATH=/usr/lib/cipher`; if the container override removes it, substrate fails to find `libc10.so`. Mitigation: end-user must include `/usr/lib/cipher` in their LD_LIBRARY_PATH.

### 3.2 cipher_kmod fails to load on a host

Symptoms: `cipher-platform status` reports `kmod loaded: False`; `/dev/cipher` absent.

Diagnose:

```sh
dkms status cipher-kmod              # should show installed
dmesg | tail -50                     # look for kmod load errors
modinfo cipher_kmod                  # confirm 0.6.5 build present
journalctl -k --since '1 hour ago' | grep cipher  # kernel log
```

Common causes:
- DKMS build failed during install. Fix: `sudo dkms build -m cipher-kmod -v 0.6.5 --force; sudo dkms install -m cipher-kmod -v 0.6.5 --force; sudo modprobe cipher_kmod`.
- Kernel headers package missing (`linux-headers-generic` is `Recommends` not `Depends`; some neocloud images strip Recommends). Fix: `sudo apt install linux-headers-$(uname -r)`.
- Module signed but EFI Secure Boot enforcement rejects unsigned (DKMS auto-signing requires MOK enrollment). Fix: either enroll MOK or disable Secure Boot on this node.

### 3.3 CDI patch missing from /var/run/cdi/nvidia.yaml

Symptoms: `cipher-platform status` reports `CDI patch: applied=False`.

Diagnose:

```sh
sudo /usr/lib/cipher/cipher_cdi_patch.sh status --json
ls -la /var/run/cdi/nvidia.yaml      # confirm exists
journalctl -u cipher-platform-watch.service --since '1 day ago'
```

Common causes:
- `/var/run/cdi/nvidia.yaml` not yet generated (NVIDIA driver installed but `nvidia-ctk cdi generate` never ran). Fix: `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml; sudo cipher-platform refresh`.
- Watcher disabled and patch wiped by driver upgrade (see §2).
- nvidia.yaml malformed by another tool; cipher_cdi_patch.py exits with rc=2. Fix: regenerate via `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml; sudo cipher-platform refresh`.

### 3.4 systemd path-watcher failure

Symptoms: `systemctl is-active cipher-platform-watch.path` reports `failed` or `inactive`.

Diagnose:

```sh
systemctl status cipher-platform-watch.path
systemctl status cipher-platform-watch.service
journalctl -u cipher-platform-watch.service --since '1 day ago'
```

Common causes:
- StartLimitBurst exceeded (more than 5 path-change events in 10 seconds). Fix: `sudo systemctl reset-failed cipher-platform-watch.service; sudo systemctl restart cipher-platform-watch.path`.
- `/var/run/cdi/nvidia.yaml` path doesn't exist at boot (NVIDIA driver loaded but CDI not generated). The path-watcher tolerates this; first apply happens once nvidia.yaml exists and gets patched.
- Manual `cipher-platform refresh` works while watcher does not. Fix: rely on manual refresh per §2 Option B; file a ticket with CIPHER support.

## 4. Uninstall and clean-room recovery

To uninstall cipher-platform from a node:

```sh
sudo apt remove cipher-platform
```

The prerm hook performs:
- `systemctl disable --now cipher-platform-watch.path`
- `cipher_cdi_patch.sh revert` (removes CIPHER entries from `/var/run/cdi/nvidia.yaml`)
- `modprobe -r cipher_kmod` + `dkms remove cipher-kmod`
- `rm /dev/cipher` (if surviving)

Verify clean state:

```sh
which cipher-platform                 # should be empty
ls /usr/lib/cipher                    # should not exist
lsmod | grep cipher_kmod              # should be empty
ls /dev/cipher                        # should not exist
grep CIPHER /var/run/cdi/nvidia.yaml  # should be empty
```

To purge configuration as well: `sudo apt purge cipher-platform`.

## 5. CIPHER support contact

For incidents, questions, or runbook gaps, contact CIPHER support at the channel established in your design-partner agreement. When filing a ticket, include:

- `sudo cipher-platform status --json` output
- `sudo cipher-platform verify --json` output
- `journalctl -u cipher-platform-watch.service --since '1 day ago'` tail
- `dmesg | grep -i cipher` tail
- NVIDIA driver version (`nvidia-smi --query-gpu=driver_version --format=csv`) and nvidia-container-toolkit version (`dpkg -s nvidia-container-toolkit | grep Version`)
- Kernel version (`uname -r`)

If `cipher-platform` CLI is itself failing, run `/usr/lib/cipher/health-check.sh` instead (the standalone health validator); it does not depend on the CLI and can run when other parts of CIPHER are broken.

---

**Anchors at v2.0 ship (cipher-platform 2.0):**
- libcipher_rt.so md5 `1d91e7dabba6b3ace520d78fdb085fc3` (Phase A close `v1-substrate-driver-worker-init`)
- libcipher_v2.so md5 `cc0479b836e560619e2b286ca1caecb7` (Track 2 close)
- cipher_kmod 0.6.5 srcversion `C36ED68CEEDD744995C0AC4` (week-13-14-complete)
- cipher_vllm plugin chain 0.2.0 (Week 5 KV-dedup + CP 5.1 + CP 5.2)
