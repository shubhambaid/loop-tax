#!/usr/bin/env bash
# Smoke test on the remote box: reference loads on Metal, one KL-base chunk, speed, and a dry run of the scan overrides.
set -uo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"; B="$ROOT/llama.cpp/build/bin"; M="$ROOT/models/Inkling-2-Mini-NVFP4.gguf"
mkdir -p "$ROOT/work"
echo "== perplexity, 1 chunk of 2048"
/usr/bin/time -l "$B/llama-perplexity" -m "$M" -f "$ROOT/data/eval/wikitext.txt" -c 2048 --chunks 1 -ngl 99 \
  --kl-divergence-base "$ROOT/work/smoke.kld" 2>&1 | grep -E "recommendedMaxWorkingSet|model buffer size|KV buffer|compute buffer|prompt eval time|Final estimate|maximum resident|error|failed" 
ls -la "$ROOT/work/smoke.kld"
echo "== bench"
"$B/llama-bench" -m "$M" -p 512 -n 128 -r 1 -ngl 99 2>&1 | tail -4
echo "== quantize dry run: block 6 to Q2_K, everything else as reference"
"$B/llama-quantize" --dry-run --allow-requantize --pure \
  --tensor-type '^blk\.6\..*weight$=q2_k' \
  --tensor-type '^blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19)\.ffn_(gate_up|down)_exps\.weight$=nvfp4' \
  "$M" "$ROOT/work/dry.gguf" Q8_0 2>&1 | grep -E "MiB ->|override|model size|quant size|error" | head -40
