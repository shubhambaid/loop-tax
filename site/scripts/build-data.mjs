// Build-time data for the site. Every number shown on the page comes from files under ../results, ../toy and
// ../configs, read here. Nothing is typed by hand. Raw result files are copied to public/data for download.
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync, cpSync, statSync } from "node:fs";
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

// per-variant JSON details that the CSV flattens away (recipes, tensor types per block for phase 3)
const recipes = {};
const p3 = join(results, "phase3-recipes");
if (existsSync(p3)) {
  for (const f of readdirSync(p3).filter((f) => f.endsWith(".json"))) {
    const r = readJSON(join(p3, f));
    if (r.spec?.recipe) recipes[r.variant] = r.spec.recipe;
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
const env = readJSON(join(results, "env", "remote_inventory.json"));
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

const out = { generatedFrom: gitCommit, experiments, recipes, toy, models, evalset, env, layout: layoutSummary, community };
mkdirSync(join(here, "..", "src", "data"), { recursive: true });
writeFileSync(join(here, "..", "src", "data", "site.json"), JSON.stringify(out));

// downloadable raw results
const pub = join(here, "..", "public", "data");
mkdirSync(pub, { recursive: true });
cpSync(results, join(pub, "results"), { recursive: true });
console.log(`site data: ${Object.keys(experiments).join(", ")} (repo ${gitCommit})`);
