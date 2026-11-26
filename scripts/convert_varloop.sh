#!/usr/bin/env bash
# Convert the TML-d24-varloop-research safetensors (NVFP4/MXFP8 experts) to a BF16 GGUF with the pinned converter.
# The converter dequantizes the NVFP4/MXFP8 experts exactly, so BF16 holds the released values. Needs ~53 GB free.
set -euo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"
SRC="$ROOT/models/TML-d24-varloop-research-hf"
OUT="$ROOT/models/TML-d24-varloop-research-BF16-local.gguf"
[[ -f "$OUT" ]] && { echo "exists: $OUT"; exit 0; }
PYTHONPATH="$ROOT/llama.cpp/gguf-py" "$ROOT/venv/bin/python" "$ROOT/llama.cpp/convert_hf_to_gguf.py" \
  "$SRC" --outtype bf16 --outfile "$OUT.partial"
mv "$OUT.partial" "$OUT"
