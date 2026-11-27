# LoopTax

**Does a looped transformer pay its quantization error more than once? On Inkling 2 Mini: no, reuse does not compound it.**

LoopTax tests a simple hypothesis on [Inkling 2 Mini](https://huggingface.co/blog/inkling2mini) from Thinking Machines
Lab, which stores 24 decoder layers but runs 48 per token: blocks 6-17 share weights and run three times. In a loop the
same quantized weights `W + e` are applied on every pass, so the error might add up coherently instead of averaging out
(the "loop tax"). If so, bits spent on looped weights would be worth more than bits spent elsewhere.

What we found (screening runs; held-out confirmation status below):

- **No compounding.** Quantizing one block of the variable-loop sibling checkpoint (TML-d24-varloop-research) and running
  it with 2-5 loop passes, the KL from the full-precision model at the same pass count scales like N^-0.28 to N^0.22
  for looped blocks and N^-0.13 to N^-0.04 for control blocks that run once after the loop. Compounding errors would
  scale like N^1 to N^2.
- **Position, not reuse.** Looped blocks are more sensitive, but already with a single pass: the middle of the network,
  and the attention of its global-attention layers (blocks 11 and 17) in particular, is what matters.
- **Why the intuition fails here.** Every pass restarts from the same prelude output; only a message read from the
  previous pass, through a learned rank-64 gate, carries over. An error made inside a pass is not re-processed by the
  shared weights the way the toy assumes.
- **FP4-trained weights should stay FP4.** The experts were trained and released in NVFP4. The ggml-org NVFP4 GGUF keeps
  them bit for bit and sits at KL 0.0024 from the BF16 GGUF (the release upcast), against 0.0118 for a same-size
  Q4_K_M, which re-grids FP4-trained weights. This is about FP4-trained weights, not about formats in general.

All numbers come from `results/` and are traced in `notebook/LAB_NOTEBOOK.md`.

![Inkling 2 Mini looped architecture](assets/1_architecture.png)

## Status

| Stage | Status |
|---|---|
| Toy simulation (random weights) | Done, see below |
| Setup: pinned llama.cpp, reference GGUF, eval set, GGUF layout check | Done (`docs/ARCHITECTURE.md`, `results/layout/`) |
| Phase 1: uniform baselines vs NVFP4 reference, community GGUFs | Done (`results/phase1-*`) |
| BF16 reference check | Done (`results/bf16check/`) |
| Phase 2: per-block and per-family sensitivity scan | Done (`results/phase2-scan/`) |
| Headline: same block, 1-6 loop passes, varloop checkpoint | Done on screening; held-out N=1,3,5 agrees, N=2-5 running (`results/phase4-*`) |
| Phase 4: bits vs loop passes | Done (`results/phase4-passes/`) |
| Phase 3: under 16 GB (sub-NVFP4 experts, scan-guided) and 16 GB+ (NVFP4 experts kept, attention tuned) | Running |

**Held-out status.** A held-out check over 1, 3 and 5 passes agrees with the screening fit (looped blocks 9 and 17:
N^0.23 and N^0.30; control block 20: N^-0.04; `results/phase4-holdout/alpha_fit.json`). The held-out run over 2-5
passes, the statistic quoted above, is running.

![Same block, more loop passes](assets/headline_passes.png)

The running log is `notebook/LAB_NOTEBOOK.md`.

## The intuition we tested (toy simulation)

> **These are toy results.** Random pre-norm residual MLP blocks (d=256, hidden 1024), round-to-nearest symmetric per-output-channel quantization, hidden-state relative L2 error against FP. They are **not** Inkling 2 Mini measurements. Real models have outliers, learned gates between passes and NVFP4/MXFP8 source weights.

**1. Same depth, same 4-bit quantization (20 seeds).** The looped stack (4 blocks x 3 passes) ends with 17.6% error vs 13.5% for 12 unique blocks, about 30% more. The curves are identical until the loop restarts.

![Error vs depth](assets/2_toy_depth.png)

**2. Where to spend bits (Inkling-shaped toy: 24 blocks, blocks 6-17 looped 3x, 10 seeds).** At the same 4.5 average bits per weight:

| Loop bits | Edge bits | Final error |
|---|---|---|
| 5 | 4 | 14.8% |
| 4 | 5 | 22.1% |

Giving the extra bit to the loop beats giving it to the edges at every budget tested (3.5 to 7.5 bits).

![Bit budget](assets/3_toy_bit_budget.png)

### Reproduce

```bash
pip install -r requirements.txt
cd toy
python sim.py                                         # writes results.json
python -c "import json, sim; json.dump(sim.exp_grid(), open('grid.json','w'), indent=1)"  # bit-budget sweep (a few minutes on CPU)
python charts.py arch depth budget                    # writes ../assets/*.png
```

## Roadmap: the real experiment

1. **Sensitivity scan.** Quantize one layer at a time and measure output KL divergence against the reference model; compare looped (6-17) vs non-looped layers.
2. **Loop-aware quant.** Keep the looped block at higher precision and push the rest lower; compare quality per GB against uniform community GGUFs.
3. **Bits x passes.** On the variable-loop research checkpoint, grid over bit width and 1/2/3 loop passes.
4. **Release.** Publish quants on Hugging Face with the measurements.

## Related

- [Inkling 2 Mini on Hugging Face](https://huggingface.co/blog/inkling2mini)
- QDA: Closed-Loop LLM Quantization Design with Agents (arXiv 2611.23388), which uses output KL to score quantization recipes

## License

Apache 2.0
