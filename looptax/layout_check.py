"""Prove how a GGUF represents the Inkling loop: physical blocks, loop metadata, where loop tensors live,
whether any weights are duplicated per visit, and how parameters split across tensor families.

    python -m looptax.layout_check HEADER.gguf[.head] --label NAME --out results/layout/NAME.json
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import subprocess

from .gguf_header import parse


def family(suffix: str) -> str:
    if "exps" in suffix:
        return "experts"
    if suffix.startswith("loop_"):
        return "loop_ctrl"
    if "gate_inp" in suffix or "exp_probs" in suffix:
        return "router"
    if suffix.startswith("attn"):
        return "attn"
    return "other"


def check(path: str, label: str) -> dict:
    info = parse(path)
    kv = info["kv"]
    arch = kv.get("general.architecture")
    loop = {k.split(".", 1)[1]: v for k, v in kv.items() if k.startswith(f"{arch}.loop.")}
    start, end = loop.get("loop.start"), loop.get("loop.end")
    blocks = collections.defaultdict(dict)
    nonblock = {}
    params = collections.Counter()
    per_block = collections.defaultdict(collections.Counter)
    types = collections.defaultdict(collections.Counter)
    for t in info["tensors"]:
        n = 1
        for d in t["dims"]:
            n *= d
        m = re.match(r"blk\.(\d+)\.(.+)", t["name"])
        if m:
            b, suf = int(m.group(1)), m.group(2)
            blocks[b][suf] = t["type"]
            fam = family(suf)
            per_block[b][fam] += n
            types[fam][t["type"]] += 1
        else:
            nonblock[t["name"]] = t["type"]
            fam = t["name"].split(".")[0]
            types[fam][t["type"]] += 1
        params[fam] += n
    idx = sorted(blocks)
    loop_tensor_blocks = sorted(b for b, ts in blocks.items() if any(s.startswith("loop_") for s in ts))
    # duplication check: identical tensor-name sets across blocks are expected; duplication per visit would show
    # up as more physical blocks than block_count, or block indices >= block_count
    n_declared = kv.get(f"{arch}.block_count")
    in_loop = lambda b: start is not None and start <= b < end
    visits = start + (end - start) * loop.get("loop.count", 1) + (n_declared - end) if start is not None else n_declared
    loop_params = sum(sum(c.values()) for b, c in per_block.items() if in_loop(b))
    edge_params = sum(sum(c.values()) for b, c in per_block.items() if not in_loop(b))
    return {
        "label": label,
        "source_file": path.split("/")[-1].removesuffix(".head"),
        "architecture": arch,
        "block_count_declared": n_declared,
        "physical_blocks_found": len(idx),
        "block_index_range": [idx[0], idx[-1]],
        "loop_metadata": loop,
        "loop_core_blocks": list(range(start, end)) if start is not None else None,
        "blocks_with_loop_ctrl_tensors": loop_tensor_blocks,
        "layer_visits_per_token": visits,
        # duplicated weights would show as one physical block per visit (48) instead of one per stored layer (24)
        "weights_duplicated_per_visit": len(idx) == visits and visits != n_declared,
        "one_physical_block_per_stored_layer": len(idx) == n_declared and idx == list(range(n_declared)),
        "n_tensors": info["n_tensors"],
        "params_by_family": dict(params),
        "params_total": sum(params.values()),
        "params_loop_blocks": loop_params,
        "params_edge_blocks": edge_params,
        "tensor_types_by_family": {k: dict(v) for k, v in types.items()},
        "block_tensors": {str(b): blocks[b] for b in idx},
        "nonblock_tensors": nonblock,
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    r = check(a.file, a.label)
    short = {k: v for k, v in r.items() if k not in ("block_tensors", "nonblock_tensors", "tensor_types_by_family")}
    print(json.dumps(short, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
