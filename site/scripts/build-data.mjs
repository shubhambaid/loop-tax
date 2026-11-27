// Build-time data for the site. Every number shown on the page comes from files under ../results, ../toy and
// ../configs, read here. Nothing is typed by hand. Raw result files are copied to public/data for download.
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync, cpSync, statSync, rmSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";
import { csvParse, autoType } from "d3-dsv";
import YAML from "yaml";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..");
const results = join(repo, "results");
const readJSON = (p) => JSON.parse(readFileSync(p, "utf8"));

const experiments = {};
for (const d of readdirSync(results)) {
  const f = join(results, d, "summary.csv");
  if (existsSync(f)) experiments[d] = csvParse(readFileSync(f, "utf8"), autoType);
}

// Phase 3 recipe family per variant (from the generated config)
const p3family = {};
for (const f of ["phase3_sub16.yaml", "phase3_frontier.yaml"]) {
  const c = YAML.parse(readFileSync(join(repo, "configs", f), "utf8"));
  for (const v of c.variants) p3family[v.id] = v.family;
}
experiments["phase3"] = [...(experiments["phase3-sub16"] ?? []), ...(experiments["phase3-frontier"] ?? [])];
for (const r of experiments["phase3"]) r.recipe_family = p3family[r.variant];

// where each BF16-check variant ran, from its config (run.py: "--device none" is CPU only, ngl null is llama.cpp's
// --fit split, otherwise everything is offloaded to the GPU)
{
  const c = YAML.parse(readFileSync(join(repo, "configs", "bf16check.yaml"), "utf8"));
  const device = (llama = {}) => (llama.extra ?? []).join(" ").includes("--device none") ? "CPU"
    : llama.ngl === null ? "CPU+GPU split via --fit" : "GPU";
  const dev = { [c.reference.id]: device(c.reference.llama) };
  for (const v of c.variants) dev[v.id] = device({ ...c.llama, ...v.llama });
  for (const r of experiments["bf16check"] ?? []) r.device = dev[r.variant];
}

// per-variant JSON details that the CSV flattens away (recipes, tensor types per block for phase 3)
const recipes = {};
const p3 = join(results, "phase3-recipes");
if (existsSync(p3)) {
  for (const f of readdirSync(p3).filter((f) => f.endsWith(".json"))) {
    const r = readJSON(join(p3, f));
    if (r.spec?.recipe) recipes[r.variant] = r.spec.recipe;
  }
}

// headline fit and per-block weight error (normalisation for the scan)
const alphaFile = join(results, "phase4-passes", "alpha_fit.json");
const alpha = existsSync(alphaFile) ? readJSON(alphaFile) : [];
const holdoutFile = join(results, "phase4-holdout", "alpha_fit.json");
const alphaHoldout = existsSync(holdoutFile) ? readJSON(holdoutFile) : [];
const refcheckFile = join(results, "varloop-refcheck", "nvfp4r.json");
const refcheck = existsSync(refcheckFile) ? readJSON(refcheckFile).domains : null;
const werrDir = join(results, "phase2-scan-weight-error");
const werr = existsSync(werrDir) ? Object.fromEntries(readdirSync(werrDir).filter((f) => f.endsWith(".json")).map((f) => {
  const r = readJSON(join(werrDir, f)); const t = r.weight_error.block_total;
  return [r.variant, { rel_err: t.rel_err, bits_removed: t.bits_removed, attn: r.weight_error.attn?.rel_err,
    gate_up: r.weight_error.gate_up?.rel_err, down: r.weight_error.down?.rel_err }];
})) : {};
for (const row of experiments["phase2-scan"] ?? []) {
  const w = werr[row.variant];
  if (w && row.kld_mean_avg != null) {
    Object.assign(row, { rel_err: w.rel_err, bits_removed: w.bits_removed,
      kl_per_gbit_removed: w.bits_removed > 0 ? row.kld_mean_avg / (w.bits_removed / 1e9) : null,
      kl_per_rel_err: w.rel_err > 0 ? row.kld_mean_avg / w.rel_err : null });
  }
}

const toy = {
  label: "Toy simulation, random weights, not Inkling 2 Mini",
  depth: readJSON(join(repo, "toy", "results.json")).depth,
  budget: readJSON(join(repo, "toy", "results.json")).budget,
  grid: readJSON(join(repo, "toy", "grid.json")),
};

const models = YAML.parse(readFileSync(join(repo, "configs", "models.yaml"), "utf8"));
const evalset = readJSON(join(repo, "configs", "evalset_manifest.json"));
const envRaw = readJSON(join(results, "env", "remote_inventory.json"));

// Privacy scrub for everything the site publishes: the remote box's hostname and absolute home-directory paths are
// replaced (paths become relative to the work root). The repo's results/ files are left untouched.
const hostname = envRaw.hostname;
const scrub = (text) => {
  let t = text.replace(/\/Users\/[^/"'\s]+\/looptax\//g, "").replace(/\/(?:Users|home)\/[^/"'\s]+/g, "~");
  if (hostname) t = t.split(hostname).join("remote-m4pro");
  return t;
};
const { hostname: _h, ...env } = envRaw;
const layout = readJSON(join(results, "layout", "inkling2-mini-nvfp4.json"));
const layoutSummary = {
  architecture: layout.architecture, block_count: layout.block_count_declared, physical_blocks: layout.physical_blocks_found,
  loop: layout.loop_metadata, visits: layout.layer_visits_per_token, params_total: layout.params_total,
  params_by_family: layout.params_by_family, params_loop_blocks: layout.params_loop_blocks,
  params_edge_blocks: layout.params_edge_blocks, duplicated: layout.weights_duplicated_per_visit,
};
const community = readdirSync(join(results, "layout")).filter((f) => f.startsWith("community-")).map((f) => {
  const l = readJSON(join(results, "layout", f));
  return { label: l.label.replace("community-", ""), architecture: l.architecture, block_count: l.block_count_declared,
           physical_blocks: l.physical_blocks_found, duplicated: l.weights_duplicated_per_visit };
});

let gitCommit = "unknown";
try { gitCommit = execSync("git rev-parse --short HEAD", { cwd: repo }).toString().trim(); } catch {}

const out = { generatedFrom: gitCommit, experiments, alpha, alphaHoldout, refcheck, recipes, toy, models, evalset, env, layout: layoutSummary, community };
mkdirSync(join(here, "..", "src", "data"), { recursive: true });
writeFileSync(join(here, "..", "src", "data", "site.json"), scrub(JSON.stringify(out)));

// downloadable raw results
const pub = join(here, "..", "public", "data");
mkdirSync(pub, { recursive: true });
rmSync(join(pub, "results"), { recursive: true, force: true });
const copyScrubbed = (src, dst) => {
  mkdirSync(dst, { recursive: true });
  for (const f of readdirSync(src)) {
    const s = join(src, f), d = join(dst, f);
    if (statSync(s).isDirectory()) copyScrubbed(s, d);
    else if (/\.(json|csv|md|txt|yaml|yml|log)$/.test(f)) writeFileSync(d, scrub(readFileSync(s, "utf8")));
    else cpSync(s, d);
  }
};
copyScrubbed(results, join(pub, "results"));
console.log(`site data: ${Object.keys(experiments).join(", ")} (repo ${gitCommit})`);
