#!/usr/bin/env bash
# Fetch only the first N MB of a GGUF on Hugging Face (enough for metadata + tensor table).
#   scripts/fetch_gguf_header.sh REPO REVISION FILE OUTDIR [MB]
set -euo pipefail
repo=$1 rev=$2 file=$3 out=$4 mb=${5:-32}
mkdir -p "$out"
curl -fsSL -r 0-$((mb*1024*1024-1)) -o "$out/$file.head" "https://huggingface.co/$repo/resolve/$rev/$file"
echo "$out/$file.head"
