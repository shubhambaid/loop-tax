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

for what in "$@"; do
  case "$what" in
    inkling-nvfp4)   fetch $INK $INK_REV Inkling-2-Mini-NVFP4.gguf ;;
    inkling-q8)      fetch $INK $INK_REV Inkling-2-Mini-Q8_0.gguf ;;
    inkling-iq4nl)   fetch $INK $INK_REV Inkling-2-Mini-IQ4_NL.gguf ;;
    inkling-imatrix) fetch $INK $INK_REV imatrix.gguf ;;
    varloop-q8)      fetch $VAR $VAR_REV TML-d24-varloop-research-Q8_0.gguf ;;
    varloop-iq4nl)   fetch $VAR $VAR_REV TML-d24-varloop-research-IQ4_NL.gguf ;;
    varloop-imatrix) fetch $VAR $VAR_REV imatrix-TML-d24-varloop-research.gguf ;;
    *) echo "unknown target $what"; exit 1 ;;
  esac
done
