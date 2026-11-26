#!/usr/bin/env bash
# Phase 4 / headline pipeline on goosemini, in order. Each step is resumable (skips finished outputs).
set -uo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"; PY="$ROOT/venv/bin/python"
cd "$ROOT/repo"
export LOOPTAX_LLAMA_BUILD=build-lt   # pinned commit + per-chunk sums patch (bootstrap CIs)
bash scripts/make_varloop_ref.sh || exit 1
# distance of the NVFP4R reference from the Q8_0 conversion (3 passes), then drop the Q8_0 files to free disk
$PY -m looptax.run configs/varloop_refcheck.yaml --make-base && $PY -m looptax.run configs/varloop_refcheck.yaml || exit 1
rm -f "$ROOT/models/TML-d24-varloop-research-Q8_0-local.gguf" "$ROOT"/kld/varloop-q8_0/*.kld
echo "deleted varloop Q8_0 conversion and its logits"
$PY -m looptax.run_passes configs/phase4_passes.yaml --make-base || exit 1
$PY -m looptax.run_passes configs/phase4_passes.yaml
