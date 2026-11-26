"""Phase 4 runner: bits x loop passes on the variable-loop checkpoint.

For every pass count N the reference (varloop Q8_0) produces its own logits. Each quantized variant is built once and
then loaded with `--override-kv inkling2_loop.loop.count=int:N`. Two KL numbers per (variant, N):
  same     KL vs the reference at the same N passes: the cost of quantization at that depth.
           The loop tax predicts this grows with N, because the same weight error is applied N times.
  vs3      KL vs the reference at 3 passes (the default): total distance from the model as shipped.
Speed per (variant, N) comes from llama-completion timings (llama-bench cannot override the loop count).

    python -m looptax.run_passes configs/phase4_passes.yaml --make-base
    python -m looptax.run_passes configs/phase4_passes.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import yaml

from . import run as R

KV = "inkling2_loop.loop.count=int:{}"


def base(cfg, domain, n):
    return R.rp(cfg.get("kld_dir", "kld")) / cfg["reference"]["id"] / f"{domain}-c{cfg['eval']['ctx']}-p{n}.kld"


def ngl(spec):
    v = spec.get("llama", {}).get("ngl", 99)
    return [] if v is None else ["-ngl", str(v)]


def make_base(cfg, dry):
    ref = cfg["reference"]
    out_dir = R.REPO / "results" / cfg["experiment"]
    log = R.rp("logs") / f"{cfg['experiment']}-reference.log"
    for n in cfg["passes"]:
        res = out_dir / f"{ref['id']}-p{n}.json"
        if res.exists():
            print(f"skip reference p{n}")
            continue
        rec = {"experiment": cfg["experiment"], "variant": ref["id"], "role": "reference", "passes": n,
               "model_file": R.rp(ref["model"]).name, **R.provenance(), "domains": {}}
        for domain, chunks in cfg["eval"]["domains"].items():
            bp = base(cfg, domain, n)
            bp.parent.mkdir(parents=True, exist_ok=True)
            cmd = [R.BIN / "llama-perplexity", "-m", R.rp(ref["model"]), "-f", R.rp(cfg["eval"]["dir"]) / f"{domain}.txt",
                   "-c", str(cfg["eval"]["ctx"]), "--chunks", str(chunks), "--kl-divergence-base", bp,
                   "--override-kv", KV.format(n), *ngl(ref)]
            t0 = time.time()
            out = R.sh(cmd, log, dry)
            rec["domains"][domain] = {**R.parse_kl(out), "seconds": round(time.time() - t0, 1)}
        if not dry:
            out_dir.mkdir(parents=True, exist_ok=True)
            res.write_text(json.dumps(rec, indent=1) + "\n")


def speed(cfg, model, n, spec, log, dry):
    sp = cfg.get("speed")
    if not sp or dry:
        return None
    # a fixed ~pp-token prompt cut from wikitext (about 0.7 words per token), then tg generated tokens
    prompt = R.rp("work") / f"speed_prompt_{sp.get('pp', 512)}.txt"
    if not prompt.exists():
        words = (R.rp(cfg["eval"]["dir"]) / "wikitext.txt").read_text().split()
        prompt.write_text(" ".join(words[: int(sp.get("pp", 512) * 0.7)]))
    cmd = [R.BIN / "llama-completion", "-m", model, "-f", prompt, "-c", str(sp.get("ctx", 1024)),
           "-n", str(sp.get("tg", 128)), "--ignore-eos", "--temp", "0", "-no-cnv", "--override-kv", KV.format(n), *ngl(spec)]
    out = R.sh(cmd, log, dry)
    res = {}
    for key, pat in (("pp_tps", r"prompt eval time\s*=.*?([\d.]+) tokens per second"),
                     ("tg_tps", r"\beval time\s*=.*?([\d.]+) tokens per second"),
                     ("pp_tokens", r"prompt eval time\s*=\s*[\d.]+ ms /\s*(\d+) tokens")):
        m = re.findall(pat, out)
        if m:
            res[key] = float(m[-1])
    return res


def run_variant(cfg, v, dry):
    out_dir = R.REPO / "results" / cfg["experiment"]
    todo = [n for n in cfg["passes"] if not (out_dir / f"{v['id']}-p{n}.json").exists()]
    if not todo:
        print(f"skip {v['id']}")
        return
    log = R.rp("logs") / f"{cfg['experiment']}-{v['id']}.log"
    t0 = time.time()
    model, made, qcmd = R.build_variant(cfg, v, log, dry)
    weights = None if dry else R.summarize_weights(model)
    for n in todo:
        rec = {"experiment": cfg["experiment"], "variant": v["id"], "role": "variant", "passes": n,
               "spec": {k: v[k] for k in v if k != "overrides"}, "reference": cfg["reference"]["id"],
               "eval": cfg["eval"], "model_file": model.name, "quantize_command": qcmd, **R.provenance(),
               "domains": {}, "domains_vs3": {}}
        for domain in cfg["eval"]["domains"]:
            for key, bn in (("domains", n), ("domains_vs3", 3)):
                if key == "domains_vs3" and (n == 3 or not v.get("vs3", True)):
                    continue
                cmd = [R.BIN / "llama-perplexity", "-m", model, "--kl-divergence-base", base(cfg, domain, bn),
                       "--kl-divergence", "-c", str(cfg["eval"]["ctx"]), "--override-kv", KV.format(n), *ngl(v)]
                t1 = time.time()
                out = R.sh(cmd, log, dry)
                rec[key][domain] = {**R.parse_kl(out), "seconds": round(time.time() - t1, 1)}
        if n == 3 and v.get("vs3", True):
            rec["domains_vs3"] = rec["domains"]
        if dry:
            continue
        rec["weights"] = weights
        rec["speed"] = speed(cfg, model, n, v, log, dry)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{v['id']}-p{n}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(f"done {v['id']} p{n}: " + ", ".join(f"{d} KLD {m.get('kld_mean')}" for d, m in rec["domains"].items()), flush=True)
    print(f"{v['id']} total {time.time() - t0:.0f}s", flush=True)
    if made and not v.get("keep", cfg.get("keep_variants", False)) and not dry:
        model.unlink()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--make-base", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(R.REPO / a.config if not os.path.isabs(a.config) else a.config))
    if a.make_base:
        make_base(cfg, a.dry_run)
        return
    for v in cfg["variants"]:
        if a.only and v["id"] not in a.only:
            continue
        run_variant(cfg, v, a.dry_run)


if __name__ == "__main__":
    main()
