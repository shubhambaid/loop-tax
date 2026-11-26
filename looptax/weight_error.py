"""Weight-space size of a perturbation: how much information a variant removes from one block, per tensor family.

For every tensor whose type differs between the reference and the variant, dequantize both (applying the per-expert
`.scale` tensor when present, as llama.cpp does at runtime) and report
  rel_err   ||W_var - W_ref||^2 / ||W_ref||^2       (relative squared weight error)
  bits_ref, bits_var, bits_removed                  (storage bits; exact, from ggml block sizes)
so that KL can be normalised per bit removed or per unit of weight error, putting blocks that start at different
precisions (NVFP4 in blocks 0-19, Q8_0 in 20-23) on the same footing.

Runs on the remote box with the llama.cpp gguf-py package (PYTHONPATH=llama.cpp/gguf-py).

    python -m looptax.weight_error REF.gguf VARIANT.gguf --block 9 --out results/.../x.json
    python -m looptax.weight_error --scan configs/phase2_scan.yaml     # regenerate each scan variant, measure, delete
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess

import numpy as np


def _family(name: str) -> str:
    if "ffn_gate_up_exps" in name:
        return "gate_up"
    if "ffn_down_exps" in name:
        return "down"
    return "attn"


def measure(ref_path: str, var_path: str, block: int) -> dict:
    from gguf import GGUFReader
    from gguf.quants import dequantize

    ref, var = GGUFReader(ref_path), GGUFReader(var_path)
    rt = {t.name: t for t in ref.tensors}
    vt = {t.name: t for t in var.tensors}
    fam = {}
    prefix = f"blk.{block}."
    for name, t in vt.items():
        if not name.startswith(prefix) or not name.endswith(".weight"):
            continue
        r = rt[name]
        if r.tensor_type == t.tensor_type:
            continue
        scale = rt.get(name[:-len(".weight")] + ".scale")
        s = np.asarray(scale.data, dtype=np.float64).reshape(-1) if scale is not None else None
        num = den = 0.0
        n_el = int(np.prod([int(x) for x in t.shape]))
        # 3D expert tensors: iterate experts to bound memory
        rd, vd = r.data, t.data
        n_exp = rd.shape[0] if rd.ndim == 3 else 1
        for e in range(n_exp):
            a = dequantize(rd[e] if rd.ndim == 3 else rd, r.tensor_type).astype(np.float64)
            b = dequantize(vd[e] if vd.ndim == 3 else vd, t.tensor_type).astype(np.float64)
            k = (s[e] if s is not None and s.size == n_exp else (s[0] if s is not None else 1.0)) ** 2
            num += k * float(((b - a) ** 2).sum())
            den += k * float((a ** 2).sum())
        f = fam.setdefault(_family(name), {"num": 0.0, "den": 0.0, "bits_ref": 0, "bits_var": 0, "numel": 0, "tensors": []})
        f["num"] += num
        f["den"] += den
        f["bits_ref"] += int(r.n_bytes) * 8
        f["bits_var"] += int(t.n_bytes) * 8
        f["numel"] += n_el
        f["tensors"].append({"name": name, "ref_type": r.tensor_type.name, "var_type": t.tensor_type.name})
    out = {}
    tot = {"num": 0.0, "den": 0.0, "bits_ref": 0, "bits_var": 0, "numel": 0}
    for k, f in fam.items():
        out[k] = {"rel_err": f["num"] / f["den"], "bits_ref": f["bits_ref"], "bits_var": f["bits_var"],
                  "bits_removed": f["bits_ref"] - f["bits_var"], "numel": f["numel"], "tensors": f["tensors"]}
        for x in tot:
            tot[x] += f[x]
    out["block_total"] = {"rel_err": tot["num"] / tot["den"] if tot["den"] else 0.0, "bits_ref": tot["bits_ref"],
                          "bits_var": tot["bits_var"], "bits_removed": tot["bits_ref"] - tot["bits_var"], "numel": tot["numel"]}
    return out


def scan(config: str, only: list[str] | None):
    import yaml

    from . import run as R

    cfg = yaml.safe_load(open(R.REPO / config))
    out_dir = R.REPO / "results" / f"{cfg['experiment']}-weight-error"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = R.rp(cfg["reference"]["model"])
    for v in cfg["variants"]:
        if v.get("block") is None or (only and v["id"] not in only):
            continue
        res = out_dir / f"{v['id']}.json"
        if res.exists():
            continue
        log = R.rp("logs") / f"{cfg['experiment']}-weight-error.log"
        tmp_cfg = {**cfg, "experiment": f"{cfg['experiment']}-werr", "work_dir": cfg.get("work_dir", "work")}
        model, made, qcmd = R.build_variant(tmp_cfg, v, log, False)
        try:
            m = measure(str(ref), str(model), v["block"])
        finally:
            if made:
                model.unlink()
        rec = {"experiment": res.parent.name, "variant": v["id"], "block": v["block"], "family": v["family"],
               "qtype": v["qtype"], "quantize_command": qcmd, **R.provenance(), "weight_error": m}
        res.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"done {v['id']}: rel_err {m['block_total']['rel_err']:.4g}, bits removed {m['block_total']['bits_removed'] / 8e9:.3f} GB", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", nargs="?")
    ap.add_argument("var", nargs="?")
    ap.add_argument("--block", type=int)
    ap.add_argument("--out")
    ap.add_argument("--scan")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    if a.scan:
        scan(a.scan, a.only)
        return
    m = measure(a.ref, a.var, a.block)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tensors"} for k, v in m.items()}, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(m, indent=1) + "\n")


if __name__ == "__main__":
    main()
