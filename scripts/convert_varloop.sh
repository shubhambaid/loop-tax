#!/usr/bin/env bash
# Convert the TML-d24-varloop-research safetensors (NVFP4/MXFP8 experts) to a Q8_0 GGUF with the pinned converter.
# The converter dequantizes the NVFP4/MXFP8 experts exactly, then writes Q8_0 (near-lossless, fits the disk budget:
# a BF16 GGUF would be 52 GB). With --delete-source the safetensors are removed after a successful conversion.
set -euo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"
SRC="$ROOT/models/TML-d24-varloop-research-hf"
OUT="$ROOT/models/TML-d24-varloop-research-Q8_0-local.gguf"
if [[ ! -f "$OUT" ]]; then
  free_gb=$(df -g "$ROOT" | awk 'NR==2{print $4}')
  (( free_gb >= 32 )) || { echo "need 32 GB free for conversion, have $free_gb"; exit 1; }
  PYTHONPATH="$ROOT/llama.cpp/gguf-py" "$ROOT/venv/bin/python" "$ROOT/llama.cpp/convert_hf_to_gguf.py" \
    "$SRC" --outtype q8_0 --outfile "$OUT.partial"
  mv "$OUT.partial" "$OUT"
fi
shasum -a 256 "$OUT"
if [[ "${1:-}" == "--delete-source" ]]; then
  rm -f "$SRC/model-00001-of-00001.safetensors"
  echo "deleted varloop safetensors"
fi
