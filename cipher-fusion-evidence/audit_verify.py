"""Op 9 AUDIT — independent offline chain verifier.

Reads the dump written by cipher_rt_audit.c's exit destructor, recomputes
the HMAC-SHA256 chain from scratch (zeros -> per-entry), and compares the
recomputed head to the live head the C side wrote. A match proves the
chain is a genuine tamper-evident HMAC chain (any altered/reordered/
dropped entry would change the recomputed head).
"""
import sys, struct, hmac, hashlib

# AUDIT_KEY — must match cipher_rt_audit.c byte-for-byte.
KEY = bytes([
    0xC1, 0x10, 0xE0, 0xA0, 0xD1, 0x70, 0xE0, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x4E, 0x45, 0x55, 0x52, 0x41, 0x4C, 0x44, 0x59,
])
ENTRY = 40  # struct cipher_rt_audit_entry

def main(path):
    with open(path, "rb") as f:
        blob = f.read()
    assert blob[:8] == b"CIPHAUD1", f"bad magic {blob[:8]!r}"
    count = struct.unpack_from("<Q", blob, 8)[0]
    live_head = blob[16:48]
    body = blob[48:]
    n = len(body) // ENTRY
    print(f"[verify] file count={count}  entries-in-ring={n}  "
          f"live head={live_head.hex()[:16]}...")

    chain = bytes(32)
    first_seq = None
    for i in range(n):
        seq, ts, delta, call_hash, op_class, decision = struct.unpack_from(
            "<QQQQIB", body, i * ENTRY)
        if first_seq is None:
            first_seq = seq
        block = (chain
                 + struct.pack("<QQQQI", seq, ts, delta, call_hash, op_class)
                 + bytes([decision]) + bytes(27))
        chain = hmac.new(KEY, block, hashlib.sha256).digest()

    recomputed = chain.hex()
    live = live_head.hex()
    full = (count == n)  # whole history is in-ring -> head must match exactly
    print(f"[verify] recomputed head = {recomputed[:16]}...")
    print(f"[verify] live head       = {live[:16]}...")
    print(f"[verify] first seq in ring = {first_seq}  full-history = {full}")

    if not full:
        print("[verify] PARTIAL: ring wrapped; cannot verify to head from "
              "zero. Re-run with fewer events (<8192).")
        return 2
    if recomputed == live:
        print(f"[verify] PASS — chain verified over {n} entries, "
              f"recomputed head == live head")
        return 0
    print("[verify] FAIL — recomputed head != live head")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/audit_op2.bin"))
