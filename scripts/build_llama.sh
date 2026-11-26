#!/usr/bin/env bash
# Build llama.cpp at the commit pinned in configs/models.yaml, with Metal. Runs on the remote box.
set -euo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"
COMMIT=$(awk '/^llama_cpp:/{f=1} f&&/commit:/{print $2; exit}' "$(dirname "$0")/../configs/models.yaml")
cd "$ROOT"
[[ -d llama.cpp ]] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git fetch -q origin && git checkout -q "$COMMIT"
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(sysctl -n hw.ncpu)" --target llama-perplexity llama-quantize llama-imatrix llama-bench llama-cli llama-gguf llama-completion
build/bin/llama-cli --version
