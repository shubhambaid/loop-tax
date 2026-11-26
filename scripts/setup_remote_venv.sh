#!/usr/bin/env bash
# Python env on the remote box: analysis deps + llama.cpp converter deps (pinned by the llama.cpp checkout).
set -euo pipefail
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"
cd "$ROOT"
[[ -d venv ]] || uv venv -p python3.12 venv
uv pip install -p venv/bin/python -r repo/requirements.txt pyarrow
uv pip install -p venv/bin/python -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
venv/bin/python -c "import torch, numpy, pyarrow, gguf; print('torch', torch.__version__, 'numpy', numpy.__version__, 'gguf', gguf.__file__)"
