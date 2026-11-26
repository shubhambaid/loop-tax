#!/usr/bin/env bash
# Overnight queue on goosemini (2026-11-26), in order; every step is resumable and validated.
#   1. headline + bits x passes grid on the varloop checkpoint (GPU)
#   2. BF16 reference check: BF16 reference logits CPU-only (--device none), then the quantized variants on the GPU
#   3. Phase 2 extension: experts-only and attention-only runs for every block
#   4. community comparison (unsloth) if downloaded
set -uo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"; PY="$ROOT/venv/bin/python"
cd "$ROOT/repo"
echo "== phase 4 $(date)"
LOOPTAX_LLAMA_BUILD=build-lt $PY -m looptax.run_passes configs/phase4_passes.yaml --make-base
LOOPTAX_LLAMA_BUILD=build-lt $PY -m looptax.run_passes configs/phase4_passes.yaml
echo "== BF16 check $(date)"
if [[ "$(shasum -a 256 "$ROOT/models/Inkling-2-Mini-BF16.gguf" | cut -d" " -f1)" == 7074c9762e6769a608c7b23960a7fd2444880b1174a3910149749f6d01389aec ]]; then
  LOOPTAX_LLAMA_BUILD=build-lt $PY -m looptax.run configs/bf16check.yaml --make-base && \
    LOOPTAX_LLAMA_BUILD=build-lt $PY -m looptax.run configs/bf16check.yaml
else
  echo "BF16 sha256 mismatch, skipped"
fi
echo "== phase 2 extension $(date)"
$PY -m looptax.run configs/phase2_scan.yaml
echo "== community $(date)"
[[ -f "$ROOT/models/Inkling-2-Mini-UD-Q4_K_M.gguf" ]] && $PY -m looptax.run configs/phase1_community.yaml --only unsloth-ud-q4_k_m
echo "== queue finished $(date)"
