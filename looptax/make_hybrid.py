"""Build the hybrid quantization source for the >= 16 GB Phase 3 frontier.

Takes every NVFP4 tensor (the original expert weights of blocks 0-19) and its `.scale` companion bit for bit from the
ggml-org NVFP4 GGUF, and every other tensor from a BF16 GGUF converted from the original release. The result keeps
the native 4-bit experts exactly and has BF16 values everywhere else, so llama-quantize can give selected tensors
(attention of the most sensitive blocks) more than Q8_0 while leaving the experts untouched.
Metadata comes from the NVFP4 file. Runs on the remote box with gguf-py from the pinned llama.cpp.

    PYTHONPATH=llama.cpp/gguf-py python -m looptax.make_hybrid NVFP4.gguf BF16.gguf OUT.gguf
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    import gguf

    ap = argparse.ArgumentParser()
    ap.add_argument("nvfp4")
    ap.add_argument("bf16")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    rn, rb = gguf.GGUFReader(a.nvfp4), gguf.GGUFReader(a.bf16)
    tb = {t.name: t for t in rb.tensors}
    arch = rn.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(a.out, arch=arch)
    for field in rn.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    w.add_key_value("looptax.hybrid_source", f"NVFP4 tensors from {a.nvfp4.split('/')[-1]}, rest from {a.bf16.split('/')[-1]}",
                    gguf.GGUFValueType.STRING)
    picked, counts = [], {"nvfp4_file": 0, "bf16_file": 0}
    for t in rn.tensors:
        from_nvfp4 = t.tensor_type == gguf.GGMLQuantizationType.NVFP4 or t.name.endswith(".scale")
        if not from_nvfp4:
            b = tb.get(t.name)
            if b is None:
                sys.exit(f"tensor {t.name} missing from the BF16 file")
            if list(b.shape) != list(t.shape):
                sys.exit(f"shape mismatch for {t.name}: {list(b.shape)} vs {list(t.shape)}")
        src = t if from_nvfp4 else tb[t.name]
        counts["nvfp4_file" if from_nvfp4 else "bf16_file"] += 1
        picked.append(src)
        w.add_tensor_info(src.name, src.data.shape, src.data.dtype, src.data.nbytes, src.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for src in picked:
        w.write_tensor_data(src.data)
    w.close()
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
