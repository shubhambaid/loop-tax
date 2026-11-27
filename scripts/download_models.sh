#!/usr/bin/env bash
# Resumable download of pinned model files into $LOOPTAX_ROOT/models. Runs on the remote box.
#   scripts/download_models.sh inkling-nvfp4 | inkling-imatrix | varloop-q8 | ...
# Revisions are pinned to the commit sha recorded in configs/models.yaml.
set -euo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"
DEST="$ROOT/models"; mkdir -p "$DEST"
TOKEN_ARGS=()
[[ -f "$HOME/.cache/huggingface/token" ]] && TOKEN_ARGS=(-H "Authorization: Bearer $(cat "$HOME/.cache/huggingface/token")")

fetch() {  # repo revision file
  local url="https://huggingface.co/$1/resolve/$2/$3"
  echo "$(date -u +%FT%TZ) fetching $1@$2 $3"
  curl -fsSL --retry 20 --retry-delay 10 -C - "${TOKEN_ARGS[@]}" -o "$DEST/$3" "$url"
  echo "$(date -u +%FT%TZ) done $3 ($(du -h "$DEST/$3" | cut -f1))"
}

INK=ggml-org/Inkling-2-Mini-GGUF;              INK_REV=0bdc580634ceac4c4d921f452777bf44228de9af
VAR=ggml-org/TML-d24-varloop-research-GGUF;    VAR_REV=7af7f5e17b57395455bd8a37d8cfefa98824ca6e

INK_HF=thinkingmachines/Inkling-2-Mini; INK_HF_REV=c9d028302cd0f035fdf8e7e91f089a4ce94d2195
INK_HF_FILES="config.json generation_config.json hf_quant_config.json chat_template.jinja model.safetensors.index.json
  processor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json tiktoken/tokenizer.model
  model-00001-of-00001.safetensors"
VAR_HF=thinkingmachines/TML-d24-varloop-research; VAR_HF_REV=91a57fe2b779bbd12c0f74d1f518cbd5b84e8d0d
VAR_HF_FILES="config.json generation_config.json hf_quant_config.json chat_template.jinja model.safetensors.index.json
  processor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json tiktoken/tokenizer.model
  model-00001-of-00001.safetensors"

fetch_dir() {  # repo revision subdir files...
  local repo=$1 rev=$2 sub=$3; shift 3
  for f in "$@"; do
    mkdir -p "$DEST/$sub/$(dirname "$f")"
    echo "$(date -u +%FT%TZ) fetching $repo@$rev $f"
    curl -fsSL --retry 20 --retry-delay 10 -C - "${TOKEN_ARGS[@]}" -o "$DEST/$sub/$f" "https://huggingface.co/$repo/resolve/$rev/$f"
  done
  echo "$(date -u +%FT%TZ) done $sub ($(du -sh "$DEST/$sub" | cut -f1))"
}

wait_for() {  # wait until no other download session is running
  while tmux has-session -t "=$1" 2>/dev/null; do sleep 60; done
}

for what in "$@"; do
  case "$what" in
    inkling-nvfp4)   fetch $INK $INK_REV Inkling-2-Mini-NVFP4.gguf ;;
    after:*)         wait_for "${what#after:}" ;;
    inkling-hf)      fetch_dir $INK_HF $INK_HF_REV Inkling-2-Mini-hf $INK_HF_FILES ;;
    varloop-hf)      fetch_dir $VAR_HF $VAR_HF_REV TML-d24-varloop-research-hf $VAR_HF_FILES ;;
    inkling-bf16)    fetch $INK $INK_REV Inkling-2-Mini-BF16.gguf ;;
    inkling-q8)      fetch $INK $INK_REV Inkling-2-Mini-Q8_0.gguf ;;
    inkling-iq4nl)   fetch $INK $INK_REV Inkling-2-Mini-IQ4_NL.gguf ;;
    inkling-imatrix) fetch $INK $INK_REV imatrix.gguf ;;
    community-ggml-iq4nl)     fetch $INK $INK_REV Inkling-2-Mini-IQ4_NL.gguf ;;
    community-unsloth-q4km)   fetch unsloth/Inkling-2-Mini-GGUF 8078d149286eec7193795413b9e0643da6d11fa1 Inkling-2-Mini-UD-Q4_K_M.gguf ;;
    community-bartowski-q4km) fetch bartowski/Inkling-2-Mini-GGUF ad66863fd3b7f3236c0b7a0862412c9e00f78a6e Inkling-2-Mini-Q4_K_M.gguf ;;
    varloop-q8)      fetch $VAR $VAR_REV TML-d24-varloop-research-Q8_0.gguf ;;
    varloop-iq4nl)   fetch $VAR $VAR_REV TML-d24-varloop-research-IQ4_NL.gguf ;;
    varloop-imatrix) fetch $VAR $VAR_REV imatrix-TML-d24-varloop-research.gguf ;;
    *) echo "unknown target $what"; exit 1 ;;
  esac
done
