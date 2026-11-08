// UI performance budgets: the built app's sizes, gzipped,
// against web/budgets.json. Run after `npm run build`:
//
//   npm run budgets
//
// "entry" is what index.html loads before the first screen: its script and
// its stylesheet. A lazy chunk loads with the screen that needs it (ECharts,
// pdf.js); the PDF worker loads only to show a PDF. The budgets were set from
// the first measurement (2026-11-04) with about 10% to spare; raising one is a
// reviewed change, like any other limit.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(web, "dist");
const budgets = JSON.parse(readFileSync(join(web, "budgets.json"), "utf8"));
const kb = (bytes) => Math.round((bytes / 1024) * 10) / 10;
const gzipped = (file) => gzipSync(readFileSync(file), { level: 9 }).length;

const html = readFileSync(join(dist, "index.html"), "utf8");
const entry = [...html.matchAll(/(?:src|href)="[^"]*\/assets\/([^"]+)"/g)].map(
  (match) => match[1],
);
const assets = readdirSync(join(dist, "assets"));
const sizes = Object.fromEntries(
  assets.map((name) => [name, gzipped(join(dist, "assets", name))]),
);
const isWorker = (name) => name.includes(".worker");
const scripts = assets.filter(
  (name) => /\.m?js$/.test(name) && !isWorker(name),
);

const measured = {
  entry_js_gzip_kb: kb(
    entry
      .filter((name) => name.endsWith(".js"))
      .reduce((sum, name) => sum + sizes[name], 0),
  ),
  entry_css_gzip_kb: kb(
    entry
      .filter((name) => name.endsWith(".css"))
      .reduce((sum, name) => sum + sizes[name], 0),
  ),
  largest_lazy_chunk_gzip_kb: kb(
    Math.max(
      0,
      ...scripts
        .filter((name) => !entry.includes(name))
        .map((name) => sizes[name]),
    ),
  ),
  total_js_gzip_kb: kb(scripts.reduce((sum, name) => sum + sizes[name], 0)),
  pdf_worker_gzip_kb: kb(
    assets.filter(isWorker).reduce((sum, name) => sum + sizes[name], 0),
  ),
  largest_asset_raw_kb: kb(
    Math.max(
      ...assets.map((name) => statSync(join(dist, "assets", name)).size),
    ),
  ),
};

let failed = false;
for (const [name, limit] of Object.entries(budgets)) {
  if (name.startsWith("$")) continue;
  const value = measured[name];
  const ok = value !== undefined && value <= limit;
  failed ||= !ok;
  console.log(
    `${ok ? "ok  " : "OVER"} ${name.padEnd(28)} ${String(value).padStart(8)} KB  (budget ${limit} KB)`,
  );
}
process.exit(failed ? 1 : 0);
