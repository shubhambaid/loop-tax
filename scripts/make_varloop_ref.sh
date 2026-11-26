#!/usr/bin/env bash
# Build the varloop reference used for the headline (reuse-count) experiment and Phase 4:
# every routed expert in all 24 blocks at NVFP4, everything else Q8_0, from our Q8_0 conversion of the release.
# All blocks start from the same precision (unlike the Inkling NVFP4 GGUF, where blocks 20-23 are Q8_0), and the
# file fits in GPU memory. Its distance from the Q8_0 conversion is measured separately (configs/varloop_refcheck.yaml).
set -euo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"; B="$ROOT/llama.cpp/build/bin"
SRC="$ROOT/models/TML-d24-varloop-research-Q8_0-local.gguf"
OUT="$ROOT/models/TML-d24-varloop-research-NVFP4R.gguf"
[[ -f "$OUT" ]] && { echo "exists: $OUT"; exit 0; }
"$B/llama-quantize" --allow-requantize --imatrix "$ROOT/models/imatrix-TML-d24-varloop-research.gguf" --pure \
  --tensor-type '^blk\.[0-9]+\.ffn_(gate_up|down)_exps\.weight$=nvfp4' "$SRC" "$OUT.partial.gguf" Q8_0 "$(sysctl -n hw.ncpu)"
mv "$OUT.partial.gguf" "$OUT"
shasum -a 256 "$OUT"
