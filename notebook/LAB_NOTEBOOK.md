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

## 2026-11-26: Phase 0 wrap-up and Phase 1, uniform baselines

**Decisions (Shubham, 2026-11-26).** Reference is the ggml-org NVFP4 GGUF. Metal wired limit raised to 21504 MB on
goosemini. Download BF16 too, as a second quantization source. Varloop: convert from the original safetensors.
Eval set as proposed. No GSM8K. ggml-org imatrix for all variants. GitHub repo `shubhambaid/loop-tax`, private for now.

**Reference checks.** `Inkling-2-Mini-NVFP4.gguf` sha256 `d4fcaf47...bd37` matches the Hugging Face LFS etag.
Loads fully on Metal (10.8 GB resident; the rest is mmapped). 259 tok/s prefill (pp512), 29 tok/s decode (tg128).

**Eval set.** `configs/evalset_manifest.json`. Context 2048, llama-perplexity scores the second half of each chunk.
wikitext-2 test: 8 chunks (8192 scored tokens). Code: CPython v3.13.0 sources, 4 chunks; at this length only the first
Python files are reached, so this domain is effectively Python stdlib code. Multilingual: FLORES-200 devtest from the
ungated `mteb/flores` copy (FLORES+ is gated and the token had not accepted its terms), 8 languages in rotating
paragraphs, 4 chunks. Reference logits: 6.5 GB on goosemini (`kld/ref-nvfp4/`).

**Pipeline control.** `identity` variant (re-quantize the reference with every tensor pinned to its own type) gives
KL = 0.000000 and 100% top-1 agreement on all three domains. llama-quantize copies a tensor byte for byte when the
target type equals the current one, so scan variants differ from the reference only where we ask.

**Phase 1 results** (`results/phase1-baselines/summary.csv`, chart `assets/phase1_kl_vs_size.png`). Source = NVFP4 GGUF,
requantized, ggml-org imatrix. Mean KL averaged over the three domains:
Q3_K_M 13.7 GB 0.0233, IQ4_XS 14.4 GB 0.0150, IQ4_NL 14.9 GB 0.0148, Q4_K_M 17.5 GB 0.0117 (0.0129 without imatrix),
Q5_K_M 19.9 GB 0.0061, Q6_K 23.5 GB 0.0037, Q8_0 27.8 GB 0.0021.

**Surprise 1: the down-projection experts cannot use K-quants.** `ffn_down_exps` has 1152 columns (the expert width),
not a multiple of 256, so llama.cpp falls back: Q2_K/Q3_K to Q4_0, Q4_K to Q5_0, Q5_K to Q5_1, Q6_K to Q8_0,
IQ4_XS to IQ4_NL. A third of all weights therefore never goes below about 4.5 bits per weight with stock types, and
the "K" recipes are larger than their names suggest (Q4_K_M is 5.36 bpw and 17.5 GB, same size as the NVFP4 file,
because its "more bits" layers push some down experts to Q6_K, which falls back to Q8_0). IQ4_XS and IQ4_NL end up
nearly identical for the same reason.

**Surprise 2 (tentative): Q4_K_M looks worse than the native NVFP4 file at the same size.** At 17.5 GB, Q4_K_M sits at
KL 0.0117 from the NVFP4 reference, while ggml-org measured the NVFP4 file at KL 0.0029 from BF16 (their number, ctx
512, different text). Comparing those two numbers is not valid: our KL is measured against NVFP4 itself, so NVFP4 scores
0 here by construction. To check it properly, re-measure NVFP4 and Q4_K_M against a BF16 reference on our eval set
once BF16 is downloaded. If it holds, re-quantizing weights trained on an FP4 grid onto a different 4-bit grid costs
real accuracy, and every loop-aware recipe must also be compared against the NVFP4 file.

**Note on llama.cpp's "more bits" heuristic.** Q4_K_M/Q5_K_M upgrade `ffn_down` for layers chosen by physical index
(`i < n/8 || i >= 7n/8 || (i - n/8) % 3 == 2`). It does not know that blocks 6-17 run three times.

**Speed (llama-bench, Metal, all layers on GPU).** pp512 / tg128 tok/s: NVFP4 259 / 29.0, IQ4_NL 260 / 30.8,
IQ4_XS 257 / 30.7, Q4_K_M 254 / 30.0, Q3_K_M 245 / 27.0, Q5_K_M 243 / 25.6. Q6_K and Q8_0 do not fit in GPU memory;
they were evaluated with partial CPU offload and not benchmarked.

## 2026-11-26: Design review (Shubham) and changes

**Position confound.** All 12 looped blocks sit in the middle of the network, so a per-block scan cannot tell "loops"
from "middle layers matter more". Fix: the headline experiment now holds position and weights fixed and changes only
the reuse count. On TML-d24-varloop-research, one block is quantized and the model is run with 1-5 loop passes; each
run is compared with the unquantized reference at the same pass count (`configs/phase4_passes.yaml`, `looptax/run_passes.py`).
Perturbed blocks: 6, 9, 12, 17 (loop) at Q2_K, block 9 also at Q4_0. Controls: blocks 3 (prelude) and 20 (coda),
which run once whatever the pass count. If their KL also rises with passes, the rise is not caused by reuse.
Prediction to test: if the error adds up coherently across passes, its amplitude grows like N and KL (quadratic in
small logit changes) like N^2; if it adds up like independent noise, amplitude grows like sqrt(N) and KL like N.

**Varloop reference.** Every routed expert in all 24 blocks at NVFP4 (requantized from our Q8_0 conversion with the
ggml-org varloop imatrix), everything else Q8_0 (`scripts/make_varloop_ref.sh`). So all blocks start from the same
precision and the reference fits in GPU memory. Its own distance from the Q8_0 conversion is measured at 3 passes
(`configs/varloop_refcheck.yaml`) before the Q8_0 is deleted for disk space.

**Blocks start from different precisions on Inkling.** In the Inkling NVFP4 reference, experts of blocks 0-19 are
NVFP4 (about 4.5 bpw) and blocks 20-23 are Q8_0, so "drop block 20 to Q2_K" removes far more information than "drop
block 9" (0.51 GB vs 0.18 GB smaller file), yet scores lower KL. Raw KL per block is not a fair comparison. Fix:
`looptax/weight_error.py` measures, for every scan variant, the bits removed and the relative squared weight error per
tensor family, by dequantizing both files. Sensitivity will be reported as KL per bit removed and per unit of weight
error as well as raw KL. First check, block 9 at Q2_K: attention rel. error 7.7% (0.020 GB removed), gate_up 7.7%
(0.159 GB), down experts (Q4_0 fallback) 0.7% with no bits removed.

**"Q2_K block" is really attention + gate_up.** Because of the 1152-column fallback, the down experts land at Q4_0 in
every scan run, about the same storage as NVFP4. The scan therefore mostly measures attention plus the gate_up
experts. The per-family runs (experts only, attention only) on 8 blocks are in the scan, and the headline experiment
adds a Q4_0 perturbation, a block-32 type that hits all three families with no fallback.

**Sample size.** 16 chunks (about 16k scored tokens) per run is screening only. Built a disjoint held-out slice
(`configs/evalset_holdout_manifest.json`: wikitext from character 300000, five other CPython files, FLORES devtest
sentences 240-1011). Candidates are rerun on it before any claim goes on the site.
