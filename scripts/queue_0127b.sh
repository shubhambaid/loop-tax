#!/usr/bin/env bash
# After the frontier: check the hybrid is release-equivalent, then confirm the frontier on held-out text (paired).
set -uo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"; PY="$ROOT/venv/bin/python"; TM=/opt/homebrew/bin/tmux
cd "$ROOT/repo"
while $TM has-session -t =q0127 2>/dev/null; do sleep 60; done
export LOOPTAX_LLAMA_BUILD=build-lt
echo "== hybrid check $(date)"
$PY -m looptax.run configs/hybrid_check.yaml
echo "== phase 3 held-out $(date)"
$PY -m looptax.run configs/phase3_holdout.yaml --make-base && $PY -m looptax.run configs/phase3_holdout.yaml && \
  rm -f "$ROOT"/kld/ref-hybrid-holdout/*.kld && echo "deleted held-out logits"
echo "== queue finished $(date)"
