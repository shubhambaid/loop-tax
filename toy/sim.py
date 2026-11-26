"""Toy simulation: does reusing a quantized layer (looping) compound error
differently from a stack of distinct quantized layers?

Model: pre-norm residual MLP blocks, h <- h + W2 @ gelu(W1 @ rmsnorm(h)).
Quantization: round-to-nearest, symmetric, per-output-channel, b bits.
Error: relative L2 distance between quantized and full-precision hidden states.
"""
import json
import numpy as np

D, H, N_TOK, SEEDS = 256, 1024, 64, 20
SCALE = 0.35


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.79788456 * (x + 0.044715 * x**3)))


def rmsnorm(x):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6)


def quant(w, bits):
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(w).max(axis=1, keepdims=True) / qmax
    return np.round(w / s).clip(-qmax, qmax) * s


def make_block(rng):
    w1 = rng.standard_normal((H, D)) / np.sqrt(D)
    w2 = rng.standard_normal((D, H)) / np.sqrt(H) * SCALE
    return w1, w2


def apply(h, blk):
    w1, w2 = blk
    return h + gelu(rmsnorm(h) @ w1.T) @ w2.T


def run(schedule, blocks, bits_for, x):
    """schedule: list of block indices; bits_for: dict idx->bits (None = FP)."""
    qblocks = {i: (blocks[i] if bits_for.get(i) is None else
                   tuple(quant(w, bits_for[i]) for w in blocks[i]))
               for i in set(schedule)}
    h_fp, h_q, errs = x.copy(), x.copy(), []
    for i in schedule:
        h_fp = apply(h_fp, blocks[i])
        h_q = apply(h_q, qblocks[i])
        errs.append(float(np.linalg.norm(h_q - h_fp) / np.linalg.norm(h_fp)))
    return errs


def exp_depth(bits=4, depth=12):
    """Same depth, same bits: 12 distinct blocks vs 4 blocks looped 3x."""
    distinct, looped = [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(s)
        x = rng.standard_normal((N_TOK, D))
        blocks = [make_block(rng) for _ in range(depth)]
        distinct.append(run(list(range(depth)), blocks,
                            {i: bits for i in range(depth)}, x))
        loop = [0, 1, 2, 3] * (depth // 4)
        looped.append(run(loop, blocks, {i: bits for i in range(4)}, x))
    return np.array(distinct), np.array(looped)


def exp_budget():
    """Inkling-shaped toy: 24 stored blocks, blocks 6-17 looped 3x (48 visits).
    Compare where to spend bits. Size = avg bits per stored weight."""
    sched = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    loop_ids, edge_ids = set(range(6, 18)), set(range(6)) | set(range(18, 24))
    configs = {
        "uniform 4-bit": {i: 4 for i in range(24)},
        "loop 4-bit, edges 6-bit": {**{i: 4 for i in loop_ids}, **{i: 6 for i in edge_ids}},
        "loop 6-bit, edges 4-bit": {**{i: 6 for i in loop_ids}, **{i: 4 for i in edge_ids}},
        "uniform 5-bit": {i: 5 for i in range(24)},
        "uniform 3-bit": {i: 3 for i in range(24)},
    }
    out = {}
    for name, cfg in configs.items():
        finals = []
        for s in range(SEEDS):
            rng = np.random.default_rng(100 + s)
            x = rng.standard_normal((N_TOK, D))
            blocks = [make_block(rng) for _ in range(24)]
            finals.append(run(sched, blocks, cfg, x)[-1])
        avg_bits = np.mean([cfg[i] for i in range(24)])
        out[name] = {"avg_bits": float(avg_bits),
                     "err_mean": float(np.mean(finals)),
                     "err_std": float(np.std(finals))}
    return out


if __name__ == "__main__":
    d, l = exp_depth()
    res = {
        "depth": {"distinct_mean": d.mean(0).tolist(), "looped_mean": l.mean(0).tolist(),
                  "distinct_std": d.std(0).tolist(), "looped_std": l.std(0).tolist()},
        "budget": exp_budget(),
    }
    json.dump(res, open("results.json", "w"), indent=1)
    print(json.dumps({"depth_final": [d.mean(0)[-1], l.mean(0)[-1]],
                      "ratio_by_layer": (l.mean(0) / d.mean(0)).round(2).tolist(),
                      "budget": res["budget"]}, indent=1))


def exp_grid(seeds=10):
    sched = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    rows = []
    data = []
    for s in range(seeds):
        rng = np.random.default_rng(200 + s)
        data.append((rng.standard_normal((N_TOK, D)), [make_block(rng) for _ in range(24)]))
    for lb in range(3, 9):
        for eb in range(3, 9):
            cfg = {i: (lb if 6 <= i < 18 else eb) for i in range(24)}
            e = [run(sched, b, cfg, x)[-1] for x, b in data]
            rows.append({"loop_bits": lb, "edge_bits": eb,
                         "avg_bits": (lb + eb) / 2, "err": float(np.mean(e))})
    return rows
