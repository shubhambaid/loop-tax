#!/usr/bin/env bash
# GPU work queue on goosemini, after the Phase 2 scan. The BF16 check has priority: it runs as soon as the BF16
# download has finished, checked between units of work. Every step is resumable.
set -uo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"; PY="$ROOT/venv/bin/python"; TM=/opt/homebrew/bin/tmux   # not $TMUX: tmux uses that env var for its socket
cd "$ROOT/repo"
while $TM has-session -t scan 2>/dev/null; do sleep 60; done
while $TM has-session -t prep_varloop 2>/dev/null; do sleep 60; done

BF16="$ROOT/models/Inkling-2-Mini-BF16.gguf"; BF16_BYTES=52226509504   # size and sha256 from the HF LFS pointer
BF16_SHA=7074c9762e6769a608c7b23960a7fd2444880b1174a3910149749f6d01389aec
bf16_ready() { [[ -f "$BF16" && $(stat -f %z "$BF16") -eq $BF16_BYTES ]] && ! $TM has-session -t dl_bf16 2>/dev/null; }
bf16_done=0
maybe_bf16() {
  if (( ! bf16_done )) && bf16_ready; then
    echo "== BF16 check $(date)"
    [[ "$(shasum -a 256 "$BF16" | cut -d" " -f1)" == "$BF16_SHA" ]] || { echo "BF16 sha256 mismatch"; bf16_done=1; return; }
    LOOPTAX_LLAMA_BUILD=build-lt $PY -m looptax.run configs/bf16check.yaml --make-base && \
      LOOPTAX_LLAMA_BUILD=build-lt $PY -m looptax.run configs/bf16check.yaml && bf16_done=1
  fi
}

maybe_bf16
export LOOPTAX_LLAMA_BUILD=build-lt
echo "== varloop reference check $(date)"
[[ -f results/varloop-refcheck/nvfp4r.json ]] || { $PY -m looptax.run configs/varloop_refcheck.yaml --make-base && $PY -m looptax.run configs/varloop_refcheck.yaml; }
# the Q8_0 conversion is only needed for the check above; free its 28 GB and the 5 GB of logits
[[ -f results/varloop-refcheck/nvfp4r.json ]] && rm -f "$ROOT/models/TML-d24-varloop-research-Q8_0-local.gguf" "$ROOT"/kld/varloop-q8_0/*.kld
maybe_bf16
export LOOPTAX_LLAMA_BUILD=build-lt
echo "== phase 4 reference logits $(date)"
$PY -m looptax.run_passes configs/phase4_passes.yaml --make-base
for v in $($PY -c "import yaml; print(' '.join(v['id'] for v in yaml.safe_load(open('configs/phase4_passes.yaml'))['variants']))"); do
  maybe_bf16
  export LOOPTAX_LLAMA_BUILD=build-lt
  echo "== phase 4 $v $(date)"
  $PY -m looptax.run_passes configs/phase4_passes.yaml --only "$v"
done
# if BF16 arrived late, wait for it
while (( ! bf16_done )); do maybe_bf16; (( bf16_done )) || sleep 120; done
# community comparison that was deprioritised behind BF16
while $TM has-session -t dl_unsloth 2>/dev/null; do sleep 60; done
$PY -m looptax.run configs/phase1_community.yaml --only unsloth-ud-q4_k_m
echo "== queue finished $(date)"
