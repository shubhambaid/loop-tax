# LoopTax lab notebook

Dated log of what ran, where, and what we learned. Newest entries at the bottom.

## 2026-11-26: Phase 0, recon and setup

**Repo.** Imported `loop-tax.bundle` (commit `1dba1e7`, toy simulation). Added `looptax/` package, `scripts/`,
`configs/`, `docs/ARCHITECTURE.md`, `Makefile`.

**Remote box (goosemini).** `results/env/remote_inventory.json`. Apple M4 Pro (12 CPU cores, 16 GPU cores, Metal 4),
24 GB unified memory, macOS 27.2, about 107 GB free disk (varies with purgeable space). No CUDA. Homebrew llama.cpp 1.0.0
(commit `b1b770d`) predates Inkling support, so we built llama.cpp from source at `07ba74b` with Metal
(`scripts/build_llama.sh`). `uv`, `tmux`, `cmake` present. MLX and PyTorch are not installed for the system Python.

**Network.** Measured on 2026-11-26: about 4.5 MB/s from goosemini to Hugging Face (500 MB range request in 110 s) and
about 4.5 MB/s laptop to goosemini over Tailscale (1 GB in 221 s, direct LAN path). The laptop gets 10-17 MB/s from HF on
short samples, but forwarding to goosemini is capped by the same link, so all downloads go straight to goosemini.
Time cost: about 65 min per 17.6 GB file, 1.7 h for a Q8_0, 3.2 h for a BF16.

**Memory constraint.** 24 GB unified memory cannot hold the BF16 (52 GB) or Q8_0 (27.8 GB) GGUF on the GPU. The
largest variant that fits is about 17-18 GB, and only if the Metal wired-memory limit allows it (default for 24 GB
machines is lower; to be measured at the smoke test).

**Model facts.** See `docs/ARCHITECTURE.md`. Key points: 24 physical blocks, blocks 6-17 run 3 times, every pass
restarts from the prelude output and receives a gated message from the previous pass, each visit has its own KV.
Experts are 93.8% of parameters and were trained with NVFP4 forward passes, so the NVFP4 expert grid is native.

**Critical check: loop representation.** Parsed the tensor tables of 7 GGUF files from 32-48 MB HTTP range requests
(`looptax/layout_check.py`, outputs in `results/layout/`). All ggml-org, unsloth, bartowski and lmstudio-community files
use `inkling2_loop` with 24 physical blocks and loop controller tensors only in blocks 6-17. One community file
(AtomicChat) declares 48 blocks under an older `inkling2` architecture but still stores 24 physical blocks; it is not used.

**Reference choice (proposed, pending Shubham's OK).** ggml-org `Inkling-2-Mini-NVFP4.gguf` (17.6 GB) as the KL
reference: it is the only near-lossless file that fits in memory, and its experts in blocks 0-19 are bit-identical to
the released checkpoint. ggml-org measured it at KL 0.0029 vs BF16, the same as Q8_0 (0.0030).

**Started.** Download of the NVFP4 reference and ggml-org `imatrix.gguf` on goosemini (tmux `dl_ref`).

**Open.** Metal memory limit, eval set, whether to also pull BF16 as the quantization source, varloop reference.
