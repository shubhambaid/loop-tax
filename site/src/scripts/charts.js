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
