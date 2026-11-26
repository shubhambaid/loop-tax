"""Parse the header of a GGUF file (metadata + tensor table) without the tensor data.

Works on a truncated file (for example the first 32 MB fetched with an HTTP range
request), so the layout of a 50 GB model can be inspected before downloading it.

Usage:
    python -m looptax.gguf_header FILE [--json OUT] [--layers]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
import sys

# ggml_type ids, from ggml/include/ggml.h. Unknown ids are printed as numbers.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
}

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.buf):
            raise EOFError("header extends past the bytes provided; fetch a larger prefix")
        (val,) = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += size
        return val

    def string(self) -> str:
        n = self.take("<Q")
        if self.pos + n > len(self.buf):
            raise EOFError("header extends past the bytes provided; fetch a larger prefix")
        s = self.buf[self.pos:self.pos + n].decode("utf-8", errors="replace")
        self.pos += n
        return s

    def value(self, vtype: int, max_array: int = 64):
        if vtype in _SCALAR:
            return self.take(_SCALAR[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            itype = self.take("<I")
            n = self.take("<Q")
            items = [self.value(itype) for _ in range(n)]
            if n > max_array:
                return {"array_len": n, "head": items[:8]}
            return items
        raise ValueError(f"unknown GGUF value type {vtype}")


def parse(path: str, prefix_mb: int = 64) -> dict:
    """Parse the header, reading only as many leading bytes as needed (doubling from prefix_mb)."""
    size = os.path.getsize(path)
    n = prefix_mb << 20
    while True:
        with open(path, "rb") as f:
            buf = f.read(n)
        try:
            return _parse_buf(buf)
        except EOFError:
            if n >= size:
                raise
            n *= 2


def _parse_buf(buf: bytes) -> dict:
    r = _Reader(buf)
    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    r.pos = 4
    version = r.take("<I")
    n_tensors = r.take("<Q")
    n_kv = r.take("<Q")
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.take("<I")
        kv[key] = r.value(vtype)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.take("<I")
        dims = [r.take("<Q") for _ in range(n_dims)]
        ttype = r.take("<I")
        offset = r.take("<Q")
        tensors.append({"name": name, "dims": dims, "type": GGML_TYPES.get(ttype, str(ttype)), "offset": offset})
    align = kv.get("general.alignment", 32)
    data_offset = (r.pos + align - 1) // align * align
    return {"version": version, "n_tensors": n_tensors, "kv": kv, "tensors": tensors, "data_offset": data_offset}


def weight_stats(path: str) -> dict:
    """Exact byte size of every tensor (needs the full file for the last tensor) and bits per weight."""
    info = parse(path)
    size = os.path.getsize(path)
    ts = sorted(info["tensors"], key=lambda t: t["offset"])
    for a, b in zip(ts, ts[1:]):
        a["nbytes"] = b["offset"] - a["offset"]
    ts[-1]["nbytes"] = size - info["data_offset"] - ts[-1]["offset"]
    n_el = sum(_numel(t) for t in ts)
    n_bytes = sum(t["nbytes"] for t in ts)
    return {"file_bytes": size, "tensor_bytes": n_bytes, "n_weights": n_el, "bpw": 8 * n_bytes / n_el,
            "tensors": {t["name"]: {"type": t["type"], "nbytes": t["nbytes"], "numel": _numel(t)} for t in ts}}


def _numel(t: dict) -> int:
    n = 1
    for d in t["dims"]:
        n *= d
    return n


def layer_summary(info: dict) -> dict:
    """Group tensors by block index: which blocks exist and what types they hold."""
    blocks = collections.defaultdict(list)
    other = []
    for t in info["tensors"]:
        m = re.match(r"blk\.(\d+)\.(.+)", t["name"])
        if m:
            blocks[int(m.group(1))].append((m.group(2), t["type"]))
        else:
            other.append((t["name"], t["type"]))
    return {"n_blocks": len(blocks), "blocks": dict(sorted(blocks.items())), "non_block": other}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--json", help="write full parsed header to this path")
    ap.add_argument("--layers", action="store_true", help="print per-block tensor list")
    a = ap.parse_args(argv)
    info = parse(a.file)
    print(f"GGUF v{info['version']}, {info['n_tensors']} tensors, {len(info['kv'])} kv")
    for k, v in info["kv"].items():
        if k.startswith("tokenizer.ggml.") and isinstance(v, dict):
            continue
        sv = str(v)
        print(f"  {k} = {sv[:120] + ' ...' if len(sv) > 120 or chr(10) in sv else sv}".replace(chr(10), " "))
    s = layer_summary(info)
    print(f"blocks: {s['n_blocks']} (indices {min(s['blocks'])}..{max(s['blocks'])})")
    types = collections.Counter(t["type"] for t in info["tensors"])
    print("tensor types:", dict(types))
    if a.layers:
        for i, ts in s["blocks"].items():
            print(f"blk.{i}: " + ", ".join(f"{n}:{ty}" for n, ty in ts))
        print("non-block: " + ", ".join(f"{n}:{ty}" for n, ty in s["non_block"]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(info, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
