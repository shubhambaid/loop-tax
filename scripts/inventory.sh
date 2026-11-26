#!/usr/bin/env bash
# Print a JSON inventory of the machine this runs on (meant for the remote box).
#   scripts/remote.sh sh bash scripts/inventory.sh > results/env/remote_inventory.json
set -uo pipefail
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
ROOT="${LOOPTAX_ROOT:-$HOME/looptax}"
LC="$ROOT/llama.cpp"
py() { python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$1"; }
mem=$(sysctl -n hw.memsize 2>/dev/null || awk '/MemTotal/{print $2*1024}' /proc/meminfo)
cat <<JSON
{
  "hostname": $(py "$(hostname)"),
  "os": $(py "$(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null) ($(uname -sr))"),
  "cpu": $(py "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"),
  "cpu_cores": $(sysctl -n hw.ncpu 2>/dev/null || nproc),
  "gpu": $(py "$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model|Total Number of Cores|Metal Support/{printf "%s; ", $2}')"),
  "memory_bytes": $mem,
  "iogpu_wired_limit_mb": $(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo null),
  "disk_free": $(py "$(df -h "$HOME" | awk 'NR==2{print $4}')"),
  "python": $(py "$(python3 --version 2>&1)"),
  "llama_cpp_commit": $(py "$(git -C "$LC" rev-parse HEAD 2>/dev/null)"),
  "llama_cpp_version": $(py "$("$LC/build/bin/llama-cli" --version 2>&1 | head -1)"),
  "llama_cpp_backend": "Metal (GGML_METAL=ON), built from source with cmake Release",
  "brew_llama_cpp": $(py "$(llama-cli --version 2>&1 | head -1) (no inkling2_loop support, unused)"),
  "mlx": $(py "$(python3 -c 'import mlx.core as m; print(m.__version__)' 2>/dev/null || echo 'not installed for system python3')"),
  "torch": $(py "$(python3 -c 'import torch; print(torch.__version__, "mps", torch.backends.mps.is_available())' 2>/dev/null || echo 'not installed for system python3')"),
  "uv": $(py "$(uv --version 2>/dev/null || echo none)"),
  "tmux": $(py "$(tmux -V 2>/dev/null || echo none)"),
  "timestamp": "$(date -u +%FT%TZ)"
}
JSON
