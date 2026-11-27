#!/usr/bin/env bash
# Queue for 2026-11-27 (after Shubham's results review). Every step is resumable and validated.
set -uo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"; PY="$ROOT/venv/bin/python"; TM=/opt/homebrew/bin/tmux
cd "$ROOT/repo"
echo "== holdout N=2,4 $(date)"
LOOPTAX_LLAMA_BUILD=build-lt $PY -m looptax.run_passes configs/phase4_holdout.yaml --make-base && \
  LOOPTAX_LLAMA_BUILD=build-lt $PY -m looptax.run_passes configs/phase4_holdout.yaml && \
  rm -f "$ROOT"/kld/varloop-nvfp4r-holdout/*.kld && echo "deleted holdout logits"
echo "== phase 3 under 16 GB $(date)"
$PY -m looptax.run configs/phase3_sub16.yaml
echo "== unsloth vs BF16 $(date)"
while $TM has-session -t dl_inkhf 2>/dev/null && [[ ! -f "$ROOT/models/Inkling-2-Mini-UD-Q4_K_M.gguf" ]]; do sleep 60; done
while $TM has-session -t dl_inkhf 2>/dev/null; do sleep 60; done
$PY -m looptax.run configs/community_vs_bf16.yaml
echo "== hybrid source $(date)"
bash scripts/convert_inkling_bf16.sh && {
  echo "== phase 3 frontier $(date)"
  $PY -m looptax.run configs/phase3_frontier.yaml
}
echo "== weight error, remaining scan variants $(date)"
PYTHONPATH="$ROOT/llama.cpp/gguf-py" $PY -m looptax.weight_error --scan configs/phase2_scan.yaml
echo "== queue finished $(date)"
