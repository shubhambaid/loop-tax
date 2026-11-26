#!/usr/bin/env bash
# Pull small result files (json/csv/log) from the remote into results/. Never pulls model weights or logit files.
set -euo pipefail
REMOTE="${LOOPTAX_REMOTE:-goosemini@100.104.128.73}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
rsync -av --prune-empty-dirs --include '*/' --include '*.json' --include '*.csv' --include '*.txt' \
  --exclude '*' "$REMOTE:looptax/results/" "$HERE/results/"
