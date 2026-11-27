// Client-side chart rendering with Observable Plot. Data comes from src/data/site.json (built from ../results).
import * as Plot from "@observablehq/plot";
import site from "../data/site.json";

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const W = (el) => Math.max(320, el.clientWidth);
const base = (el) => ({
  width: W(el), height: Math.round(Math.min(420, W(el) * 0.58)), marginLeft: 60, marginBottom: 44,
  style: { background: "transparent", color: css("--text-2"), fontSize: "12px", fontFamily: css("--font") },
});

const renderers = {
  toyDepth(el) {
    const d = site.toy.depth;
    const rows = d.distinct_mean.flatMap((v, i) => [
      { depth: i + 1, err: 100 * v, stack: "12 distinct blocks" },
      { depth: i + 1, err: 100 * d.looped_mean[i], stack: "4 blocks looped 3 times" },
    ]);
    return Plot.plot({ ...base(el),
      x: { label: "layer visits", ticks: 12 }, y: { label: "hidden-state error (%)", grid: true },
      color: { domain: ["12 distinct blocks", "4 blocks looped 3 times"], range: [css("--series-1"), css("--series-2")], legend: true },
      marks: [Plot.lineY(rows, { x: "depth", y: "err", stroke: "stack", strokeWidth: 2 }),
              Plot.dot(rows, { x: "depth", y: "err", fill: "stack", r: 4, stroke: css("--surface"), strokeWidth: 2, tip: true })] });
  },
  toyGrid(el) {
    return Plot.plot({ ...base(el), marginLeft: 70,
      x: { label: "edge bits", type: "band" }, y: { label: "loop bits", type: "band", reverse: true },
      color: { scheme: "blues", label: "final error", legend: true, tickFormat: (v) => `${(100 * v).toFixed(0)}%` },
      marks: [Plot.cell(site.toy.grid, { x: "edge_bits", y: "loop_bits", fill: "err", inset: 1, tip: true }),
              Plot.text(site.toy.grid, { x: "edge_bits", y: "loop_bits", text: (d) => `${(100 * d.err).toFixed(1)}`, fill: (d) => d.err > 0.3 ? "white" : "black", fontSize: 11 })] });
  },
  phase1(el) {
    const ours = (site.experiments["phase1-baselines"] || []).filter((r) => r.role === "variant");
    const comm = (site.experiments["phase1-community"] || []).filter((r) => r.role === "variant");
    const rows = [...ours.map((r) => ({ ...r, source: "uniform, made by us" })), ...comm.map((r) => ({ ...r, source: "community GGUF" }))];
    return Plot.plot({ ...base(el),
      x: { label: "file size (GB)", grid: true }, y: { label: "mean KL vs NVFP4 reference", type: "log", grid: true, ticks: 4, tickFormat: (v) => v.toPrecision(1) },
      color: { domain: ["uniform, made by us", "community GGUF"], range: [css("--series-1"), css("--series-2")], legend: true },
      marks: [Plot.ruleX([17.628], { stroke: css("--text-2"), strokeDasharray: "4,3" }),
              Plot.text([{ x: 17.628 }], { x: "x", frameAnchor: "top", dx: 4, textAnchor: "start", text: () => "NVFP4 reference", fill: css("--text-2") }),
              Plot.dot(rows, { x: "file_gb", y: "kld_mean_avg", fill: "source", r: 6, stroke: css("--surface"), strokeWidth: 2,
                channels: { variant: "variant", "top-1 %": "wikitext_same_top_p_pct" }, tip: true }),
              Plot.text(rows, { x: "file_gb", y: "kld_mean_avg", text: "variant", dx: 8, textAnchor: "start", fill: css("--text"), fontSize: 11 })] });
  },
  phase2(el) {
    const rows = (site.experiments["phase2-scan"] || []).filter((r) => r.family === "block" && r.kld_mean_avg != null);
    if (!rows.length) return null;
    const r2 = rows.map((r) => ({ ...r, role: r.block >= 6 && r.block <= 17 ? "looped (runs 3x)" : "runs once" }));
    return Plot.plot({ ...base(el),
      fx: { label: null }, x: { label: "physical block", type: "band" }, y: { label: "mean KL vs reference", grid: true },
      color: { domain: ["runs once", "looped (runs 3x)"], range: [css("--edge"), css("--loop")], legend: true },
      facet: { data: r2, y: "qtype", label: null },
      marks: [Plot.barY(r2, { x: "block", y: "kld_mean_avg", fill: "role", rx: 2, tip: true }),
              Plot.ruleY([0], { stroke: css("--border") })] });
  },
  headlineKL(el) {
    const rows = (site.experiments["phase4-passes"] || []).filter((r) => r.role === "variant" && r.block != null && r.kld_mean_avg != null);
    if (!rows.length) return null;
    const cat = (b) => (b < 6 ? "prelude (injected into every pass)" : b <= 17 ? "loop (reused every pass)" : "coda (runs once, control)");
    const r2 = rows.map((r) => ({ ...r, category: cat(r.block), extrapolated: r.passes > 5 }));
    const trained = r2.filter((r) => !r.extrapolated), extra = r2.filter((r) => r.extrapolated);
    return Plot.plot({ ...base(el),
      fx: { label: null, domain: ["Q2_K", "Q4_0"] },
      x: { label: "loop passes N", type: "log", ticks: [1, 2, 3, 4, 5, 6], tickFormat: (d) => (d === 6 ? "6*" : `${d}`) },
      y: { label: "KL vs reference at the same N", type: "log", grid: true, tickFormat: (v) => v.toPrecision(1) },
      color: { domain: ["loop (reused every pass)", "prelude (injected into every pass)", "coda (runs once, control)"],
               range: [css("--loop"), css("--series-4"), css("--edge")], legend: true },
      marks: [Plot.line(trained, { fx: "qtype", x: "passes", y: "kld_mean_avg", stroke: "category", z: "variant", strokeWidth: 2, sort: "passes" }),
              Plot.dot(trained, { fx: "qtype", x: "passes", y: "kld_mean_avg", fill: "category", r: 4, stroke: css("--surface"), strokeWidth: 2, channels: { variant: "variant" }, tip: true }),
              Plot.dot(extra, { fx: "qtype", x: "passes", y: "kld_mean_avg", stroke: "category", r: 4, strokeWidth: 1.5, channels: { variant: "variant", note: () => "N = 6 was never trained" }, tip: true })] });
  },
  alpha(el) {
    const label = { loop: "loop (reused every pass)", prelude: "prelude (injected into every pass)", coda: "coda (runs once, control)" };
    const tag = (rs, set) => (rs || []).filter((r) => r.alpha_2to5 != null).map((r) => ({ ...r, set, cat: label[r.category] ?? r.category,
      lo: r.alpha_2to5_ci95[0], hi: r.alpha_2to5_ci95[1], row: `${r.variant} (${set})` }));
    const r2 = [...tag(site.alpha, "screening"), ...tag(site.alphaHoldout, "held-out")]
      .sort((a, b) => a.block - b.block || a.qtype.localeCompare(b.qtype) || (a.set === "held-out") - (b.set === "held-out"));
    if (!r2.length) return null;
    const held = r2.filter((r) => r.set === "held-out"), scr = r2.filter((r) => r.set === "screening");
    return Plot.plot({ ...base(el), marginLeft: 170, height: 50 + 24 * r2.length,
      x: { label: "alpha in KL ~ N^alpha, passes 2-5 (95% bootstrap CI)", grid: true, domain: [-0.5, 2.1] },
      y: { label: null, domain: r2.map((r) => r.row) },
      color: { domain: Object.values(label), range: [css("--loop"), css("--series-4"), css("--edge")], legend: true },
      marks: [Plot.ruleX([0, 1, 2], { stroke: css("--text-2"), strokeOpacity: 0.5, strokeDasharray: "3,3" }),
              Plot.text([{ x: 1, t: "independent" }, { x: 2, t: "coherent" }], { x: "x", text: "t", frameAnchor: "top", dy: -14, fill: css("--text-2"), fontSize: 11 }),
              Plot.ruleY(r2, { y: "row", x1: "lo", x2: "hi", stroke: "cat", strokeWidth: 2 }),
              Plot.dot(scr, { y: "row", x: "alpha_2to5", fill: "cat", r: 5, stroke: css("--surface"), strokeWidth: 2,
                channels: { text: "set", "95% CI": (d) => `${d.lo.toFixed(2)} to ${d.hi.toFixed(2)}` }, tip: true }),
              Plot.dot(held, { y: "row", x: "alpha_2to5", stroke: "cat", fill: css("--surface"), r: 5, strokeWidth: 2,
                channels: { text: "set", "95% CI": (d) => `${d.lo.toFixed(2)} to ${d.hi.toFixed(2)}` }, tip: true })] });
  },
  family(el) {
    const names = { attn: "attention only", experts: "experts only" };
    const rows = (site.experiments["phase2-scan"] || []).filter((r) => names[r.family] && r.qtype === "Q2_K" && r.kld_mean_avg != null)
      .map((r) => ({ ...r, part: names[r.family], role: r.block >= 6 && r.block <= 17 ? "looped (runs 3x)" : "runs once" }));
    if (!rows.length) return null;
    return Plot.plot({ ...base(el),
      x: { label: "physical block", type: "band" }, y: { label: "mean KL vs reference", grid: true },
      fy: { label: null, domain: Object.values(names) },
      color: { domain: ["runs once", "looped (runs 3x)"], range: [css("--edge"), css("--loop")], legend: true },
      marks: [Plot.barY(rows, { fy: "part", x: "block", y: "kld_mean_avg", fill: "role", rx: 2, tip: true }),
              Plot.ruleY([0], { stroke: css("--border") })] });
  },
  bf16(el) {
    const rows = (site.experiments["bf16check"] || []).filter((r) => r.role === "variant" && r.kld_mean_avg != null)
      .map((r) => ({ ...r, kind: r.variant === "ggml-org-nvfp4" ? "NVFP4 GGUF (original experts)" : "other quants" }));
    if (!rows.length) return null;
    return Plot.plot({ ...base(el), marginLeft: 150, height: 40 + 34 * rows.length,
      x: { label: "mean KL vs BF16 (3 domains)", grid: true },
      y: { label: null, domain: rows.slice().sort((a, b) => a.kld_mean_avg - b.kld_mean_avg).map((r) => r.variant) },
      color: { domain: ["NVFP4 GGUF (original experts)", "other quants"], range: [css("--loop"), css("--edge")], legend: true },
      marks: [Plot.barX(rows, { y: "variant", x: "kld_mean_avg", fill: "kind", rx: 2, channels: { "size (GB)": "file_gb" }, tip: true }),
              Plot.text(rows, { y: "variant", x: "kld_mean_avg", text: (d) => `${d.file_gb.toFixed(1)} GB`, dx: 6, textAnchor: "start", fill: css("--text-2"), fontSize: 11 })] });
  },
  grid(el) {
    const names = { reference: "NVFP4 (reference)", "uniform-iq4_xs": "IQ4_XS experts", "uniform-q3_k": "Q3_K experts", "uniform-q2_k": "Q2_K experts" };
    const rows = (site.experiments["phase4-passes"] || []).filter((r) => names[r.variant] && r.passes <= 5)
      .map((r) => ({ ...r, series: names[r.variant], vs3: ((r.wikitext_vs3_kld_mean ?? 0) + (r.code_vs3_kld_mean ?? 0)) / 2 }))
      .filter((r) => r.vs3 > 0);
    if (!rows.length) return null;
    return Plot.plot({ ...base(el),
      x: { label: "decode tokens/s (M4 Pro)", grid: true }, y: { label: "KL vs the shipped model (reference at 3 passes)", type: "log", grid: true },
      color: { domain: Object.values(names), range: [css("--text"), css("--series-1"), css("--series-2"), css("--series-3")], legend: true },
      marks: [Plot.line(rows, { x: "tg_tps", y: "vs3", stroke: "series", sort: "passes", strokeWidth: 2 }),
              Plot.dot(rows, { x: "tg_tps", y: "vs3", fill: "series", r: 5, stroke: css("--surface"), strokeWidth: 2, channels: { passes: "passes", "file GB": "file_gb" }, tip: true }),
              Plot.text(rows, { x: "tg_tps", y: "vs3", text: (d) => `${d.passes}p`, dx: 7, dy: -6, fill: css("--text-2"), fontSize: 10 })] });
  },
  phase3(el) {
    const names = { uniform: "same type in every block", "loop-hi": "loop NVFP4, prelude + coda Q2_K",
                    "edge-hi": "prelude + coda NVFP4, loop Q2_K", guided: "guided by the experts scan",
                    frontier: "NVFP4 experts kept, rest tuned" };
    const key = "ref-bf16_kld_mean_avg";
    const rows = (site.experiments["phase3"] || []).filter((r) => r.role === "variant" && names[r.recipe_family] && r[key] != null)
      .map((r) => ({ ...r, fam: names[r.recipe_family], y: r[key] }));
    if (!rows.length) return null;
    const ctxNames = ["ggml-org-nvfp4", "ggml-org-iq4_nl", "q4_k_m-from-nvfp4", "iq4_nl-from-nvfp4"];
    const ctx = [...(site.experiments["bf16check"] || []).filter((r) => ctxNames.includes(r.variant)).map((r) => ({ ...r, y: r.kld_mean_avg })),
                 ...(site.experiments["community-vs-bf16"] || []).filter((r) => r.role === "variant" && r[key] != null).map((r) => ({ ...r, y: r[key] }))];
    return Plot.plot({ ...base(el),
      x: { label: "file size (GB)", grid: true }, y: { label: "mean KL vs BF16", type: "log", grid: true, tickFormat: (v) => v.toPrecision(1) },
      color: { domain: Object.values(names), range: [css("--text"), css("--loop"), css("--edge"), css("--series-3"), css("--series-4")], legend: true },
      marks: [Plot.dot(ctx, { x: "file_gb", y: "y", symbol: "square", stroke: css("--text-2"), r: 4, channels: { variant: "variant" }, tip: true }),
              Plot.text(ctx, { x: "file_gb", y: "y", text: "variant", dx: 6, textAnchor: "start", fill: css("--text-2"), fontSize: 10 }),
              Plot.line(rows.filter((r) => ["uniform", "guided", "frontier"].includes(r.recipe_family)), { x: "file_gb", y: "y", stroke: "fam", sort: "file_gb", strokeWidth: 2 }),
              Plot.dot(rows, { x: "file_gb", y: "y", fill: "fam", r: 5, symbol: (d) => (["loop-hi", "edge-hi"].includes(d.recipe_family) ? "diamond" : "circle"),
                stroke: css("--surface"), strokeWidth: 2, channels: { variant: "variant", "KL vs NVFP4 file": "kld_mean_avg" }, tip: true })] });
  },
  phase2norm(el) {
    const rows = (site.experiments["phase2-scan"] || []).filter((r) => r.family === "block" && r.kl_per_gbit_removed != null);
    if (!rows.length) return null;
    const r2 = rows.map((r) => ({ ...r, role: r.block >= 6 && r.block <= 17 ? "looped (runs 3x)" : "runs once" }));
    return Plot.plot({ ...base(el),
      x: { label: "physical block", type: "band" }, y: { label: "mean KL per Gbit removed", grid: true },
      color: { domain: ["runs once", "looped (runs 3x)"], range: [css("--edge"), css("--loop")], legend: true },
      facet: { data: r2, y: "qtype", label: null },
      marks: [Plot.barY(r2, { x: "block", y: "kl_per_gbit_removed", fill: "role", rx: 2, tip: true }), Plot.ruleY([0], { stroke: css("--border") })] });
  },
};

function renderAll() {
  for (const el of document.querySelectorAll("[data-chart]")) {
    const fn = renderers[el.dataset.chart];
    if (!fn) continue;
    const node = fn(el);
    el.replaceChildren(...(node ? [node] : []));
    if (!node) el.innerHTML = '<div class="pending">Not run yet. This chart fills in when the results exist in the repo.</div>';
  }
}

renderAll();
let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(renderAll, 150); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
document.addEventListener("themechange", renderAll);
