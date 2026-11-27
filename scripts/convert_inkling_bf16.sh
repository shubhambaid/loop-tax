#!/usr/bin/env bash
# Inkling 2 Mini safetensors (release c9d0283) -> BF16 GGUF with the pinned converter (NVFP4/MXFP8 experts dequantized
# exactly), then the hybrid source for the >= 16 GB frontier (looptax.make_hybrid). Deletes the safetensors after the
# conversion and the local BF16 GGUF after the hybrid is built. Needs ~75 GB free at the peak.
set -euo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"
SRC="$ROOT/models/Inkling-2-Mini-hf"
BF16="$ROOT/models/Inkling-2-Mini-BF16-local.gguf"
HYB="$ROOT/models/Inkling-2-Mini-hybrid.gguf"
[[ -f "$HYB" ]] && { echo "exists: $HYB"; exit 0; }
if [[ ! -f "$BF16" ]]; then
  PYTHONPATH="$ROOT/llama.cpp/gguf-py" "$ROOT/venv/bin/python" "$ROOT/llama.cpp/convert_hf_to_gguf.py" \
    "$SRC" --outtype bf16 --outfile "$BF16.partial"
  mv "$BF16.partial" "$BF16"
  rm -f "$SRC/model-00001-of-00001.safetensors"
fi
cd "$ROOT/repo"
PYTHONPATH="$ROOT/llama.cpp/gguf-py" "$ROOT/venv/bin/python" -m looptax.make_hybrid \
  "$ROOT/models/Inkling-2-Mini-NVFP4.gguf" "$BF16" "$HYB.partial"
mv "$HYB.partial" "$HYB"
shasum -a 256 "$HYB"
rm -f "$BF16"
