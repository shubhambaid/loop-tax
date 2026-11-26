"""Mixed-precision recipes for Inkling 2 Mini, expressed per block and per tensor family, with exact size estimates.

A recipe maps (block, family) -> ggml type, plus types for token_embd and output. It is turned into llama-quantize
--tensor-type overrides on top of `--pure Q8_0`, so nothing is left to llama.cpp's heuristics. Families:
  gate_up  blk.N.ffn_gate_up_exps.weight   (2304 columns, any type)
  down     blk.N.ffn_down_exps.weight      (1152 columns: block-256 types fall back, see FALLBACK)
  attn     every other quantizable blk.N.* weight (attention projections)
A type of "keep" leaves the tensor exactly as in the reference (NVFP4 experts in blocks 0-19 are copied bit for bit).

Size estimates use the reference tensor table (configs/tensors_inkling2_mini_nvfp4.json) and ggml block sizes.
"""

from __future__ import annotations

import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
TABLE = REPO / "configs" / "tensors_inkling2_mini_nvfp4.json"

# (block size, bytes per block) from ggml
TYPE_SIZE = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "Q5_1": (32, 24), "Q5_0": (32, 22),
    "Q4_1": (32, 20), "Q4_0": (32, 18), "IQ4_NL": (32, 18), "NVFP4": (16, 9),
    "Q6_K": (256, 210), "Q5_K": (256, 176), "Q4_K": (256, 144), "Q3_K": (256, 110), "Q2_K": (256, 84),
    "IQ4_XS": (256, 136), "IQ3_S": (256, 110), "IQ3_XXS": (256, 98), "IQ2_S": (256, 82), "IQ2_XS": (256, 74),
    "IQ2_XXS": (256, 66),
}
# llama.cpp fallback when the row length is not a multiple of the block size (src/llama-quant.cpp fallback_type_for)
FALLBACK = {"Q2_K": "Q4_0", "Q3_K": "Q4_0", "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0",
            "IQ4_XS": "IQ4_NL", "IQ3_S": "IQ4_NL", "IQ3_XXS": "IQ4_NL", "IQ2_S": "IQ4_NL", "IQ2_XS": "IQ4_NL",
            "IQ2_XXS": "IQ4_NL"}
N_BLOCKS = 24
LOOP = range(6, 18)
EDGES = [b for b in range(N_BLOCKS) if b not in LOOP]


def load_table() -> list[dict]:
    return json.loads(TABLE.read_text())["tensors"]


def family_of(name: str) -> tuple[int | None, str]:
    m = re.match(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return None, name.split(".")[0]           # token_embd / output / ...
    b, suf = int(m.group(1)), m.group(2)
    if suf.startswith("ffn_gate_up_exps.weight"):
        return b, "gate_up"
    if suf.startswith("ffn_down_exps.weight"):
        return b, "down"
    return b, "attn"


def effective_type(t: str, ncols: int) -> str:
    bs = TYPE_SIZE[t][0]
    return FALLBACK.get(t, t) if ncols % bs else t


def nbytes(t: str, dims: list[int]) -> int:
    bs, by = TYPE_SIZE[t]
    n = 1
    for d in dims:
        n *= d
    return n // bs * by


def resolve(recipe: dict, tensor: dict) -> str:
    """Type a tensor ends up with under `recipe` (reference type if not quantizable or 'keep')."""
    if tensor["type"] == "F32" or not tensor["name"].endswith(".weight"):
        return tensor["type"]
    b, fam = family_of(tensor["name"])
    want = recipe.get("output" if fam == "output" else "embd" if fam == "token_embd" else (b, fam), "keep")
    if want == "keep":
        return tensor["type"]
    return effective_type(want, tensor["dims"][0])


def size_bytes(recipe: dict, table: list[dict] | None = None) -> int:
    table = table or load_table()
    # tensors that are not weights (.scale etc) keep their reference size
    return sum(nbytes(resolve(recipe, t), t["dims"]) for t in table)


def overrides(recipe: dict, table: list[dict] | None = None) -> list[str]:
    """--tensor-type list that realises the recipe on top of `--pure` with base Q8_0."""
    table = table or load_table()
    out = []
    for t in table:
        if t["type"] == "F32" or not t["name"].endswith(".weight"):
            continue
        typ = resolve(recipe, t)
        if typ == "Q8_0" and t["type"] == "Q8_0":
            continue                                  # base type already
        out.append("^" + re.escape(t["name"]) + "$=" + typ.lower())
    return out


def make(default: dict, per_block: dict | None = None, embd="keep", output="keep") -> dict:
    """default: {family: type} for every block; per_block: {block: {family: type}} overrides."""
    r = {(b, f): default.get(f, "keep") for b in range(N_BLOCKS) for f in ("gate_up", "down", "attn")}
    for b, fams in (per_block or {}).items():
        for f, t in fams.items():
            r[(b, f)] = t
    r["embd"], r["output"] = embd, output
    return r


def describe(recipe: dict) -> dict:
    """JSON-friendly per-block type table for result files and the site."""
    return {"blocks": {str(b): {f: recipe[(b, f)] for f in ("gate_up", "down", "attn")} for b in range(N_BLOCKS)},
            "embd": recipe["embd"], "output": recipe["output"]}
