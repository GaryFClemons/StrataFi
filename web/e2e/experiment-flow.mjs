// The experiments browser flow: write an experiment in YAML, check its plan,
// save it, start it, and once its trials have finished, open every results
// view; confirm a trial on the locked test split, see a
// second unlogged read refused and a logged one go through; compare two
// trials; then promote: export the confirmed trial as a knowledge profile,
// download it, and approve it as an owner. No query text may reach the
// console.
//
//   node e2e/experiment-flow.mjs <app URL> <experiment YAML file> <trials>
//
// services/worker/tests/test_experiments_integration.py prepares the
// Halcyon demo corpus, its base documents and eval set, serves the built app
// beside the API, and runs this. STRATAFI_SCREENSHOTS=<dir> saves a
// screenshot after each step.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { signIn, switchUser } from "./lib.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [appUrl, yamlFile, trialsText] = process.argv.slice(2);
if (!appUrl || !yamlFile || !trialsText) {
  console.error(
    "usage: node e2e/experiment-flow.mjs <app URL> <experiment YAML> <trials>",
  );
  process.exit(2);
}
const trials = Number(trialsText);

const browser = await chromium.launch(
  process.env.STRATAFI_CHROMIUM
    ? { executablePath: process.env.STRATAFI_CHROMIUM }
    : {},
);
const console_ = [];
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(30_000);
  page.on("console", (message) => console_.push(message.text()));
  page.on("pageerror", (error) => console_.push(String(error)));
  // The standalone host confirms with the browser's own dialog.
  page.on("dialog", (dialog) => void dialog.accept());

  let shots = 0;
  const step = async (name) => {
    if (process.env.STRATAFI_SCREENSHOTS && shots > 0) {
      await page.screenshot({
        path: path.join(
          process.env.STRATAFI_SCREENSHOTS,
          `experiment-${shots}.png`,
        ),
        fullPage: true,
      });
    }
    shots += 1;
    process.stdout.write(`- ${name}\n`);
  };

  await step("sign in as an editor");
  await signIn(page, appUrl);

  await step("write the experiment and check its plan");
  await page.getByRole("link", { name: "Experiments", exact: true }).click();
  await page
    .getByLabel("Experiment (YAML, stratafi.experiment.v1)")
    .fill(readFileSync(yamlFile, "utf-8"));
  await page.getByRole("button", { name: "Check plan" }).click();
  const work = page.getByRole("table", { name: "Unique work" });
  await work.waitFor();
  await work.getByRole("cell", { name: "embed", exact: true }).waitFor();
  assert.match(
    await page.getByTestId("plan-summary").first().innerText(),
    new RegExp(`${trials} trials`),
  );

  await step("save it, then start it");
  await page.getByRole("button", { name: "Save for review" }).click();
  await page.waitForURL(/\/experiments\/[0-9a-f-]{36}$/);
  const experimentUrl = page.url();
  await page.getByRole("button", { name: "Start", exact: true }).click();

  await step("wait for the trials");
  await page
    .getByTestId("experiment-progress")
    .getByText(`${trials} succeeded`)
    .waitFor({ timeout: 240_000 });

  const views = page.getByRole("group", { name: "Views" });
  const view = async (name) => {
    await views.getByRole("button", { name, exact: true }).click();
  };
  const drawn = async (testId, count) => {
    const charts = page.getByTestId(testId);
    await charts.first().locator("svg").waitFor();
    assert.equal(await charts.count(), count, testId);
  };

  await step("the leaderboard");
  const board = page.getByRole("table", { name: "Leaderboard" });
  await board.waitFor();
  await page.waitForFunction(
    (count) =>
      document.querySelectorAll('table[aria-label="Leaderboard"] tbody tr')
        .length === count,
    trials,
  );

  await step("factor effects");
  await view("Factor effects");
  await drawn("effects-chart", 2);
  await page
    .getByRole("table", { name: /^Effects of / })
    .first()
    .waitFor();

  await step("interactions");
  await view("Interactions");
  await drawn("interaction-heatmap", 1);

  await step("file types");
  await view("File types");
  await page
    .getByRole("table", { name: "File types" })
    .getByRole("cell", { name: "text/markdown", exact: true })
    .waitFor();

  await step("Pareto charts");
  await view("Pareto");
  await drawn("pareto-chart", 2);

  await step("where queries were lost");
  await view("Where lost");
  await drawn("where-lost-chart", 1);
  await page
    .getByRole("table", { name: "Queries most trials missed" })
    .waitFor();

  await step("confirm a trial on the test split, once");
  await view("Leaderboard");
  await page.getByTestId("sample-size").waitFor();
  const readsTable = page.getByRole("table", { name: "Test-split reads" });
  const readRows = readsTable.locator("tbody tr");
  await page.getByRole("button", { name: "Confirm trial 0 on test" }).click();
  await readRows.first().waitFor();
  await readsTable
    .locator('[data-status="succeeded"]')
    .first()
    .waitFor({ timeout: 120_000 });
  assert.match(
    await page.getByTestId("test-reads").innerText(),
    /read 1 time\b/,
  );

  await step("a second read without a reason is refused");
  await page.getByRole("button", { name: "Confirm trial 1 on test" }).click();
  const again = page.getByRole("form", { name: "Read the test split again" });
  await again.getByRole("alert").waitFor();
  assert.equal(await readRows.count(), 1);
  assert.match(
    await page.getByTestId("test-reads").innerText(),
    /read 1 time\b/,
  );

  await step("with a reason, it's a logged second read");
  await again.getByLabel("Reason").selectOption("candidate_changed");
  await again.getByRole("button", { name: "Read again (logged)" }).click();
  await page.waitForFunction(
    () =>
      document.querySelectorAll('table[aria-label="Test-split reads"] tbody tr')
        .length === 2,
  );
  await readsTable.getByRole("cell", { name: "candidate changed" }).waitFor();
  await page.getByTestId("test-reads").getByText("read 2 times").waitFor();

  await step("compare two trials");
  await view("Leaderboard");
  await page.getByLabel("Compare trial 0").check();
  await page.getByLabel("Compare trial 1").check();
  await page
    .getByRole("link", { name: "Compare the two trials query by query" })
    .click();
  await page.waitForURL(/\/evals\/compare\/[0-9a-f-]{36}\/[0-9a-f-]{36}$/);
  await page.getByTestId("compare-delta").first().waitFor();

  await step("promote: export the confirmed trial as a profile");
  await page.goto(experimentUrl);
  await page
    .getByRole("button", { name: "Export read 1 as a profile" })
    .click();
  const exporting = page.getByRole("form", {
    name: "Export a knowledge profile",
  });
  await exporting.getByLabel("Profile name").fill("kp-browser");
  await exporting.getByRole("button", { name: "Export the profile" }).click();
  await page.waitForURL(/\/profiles\/[0-9a-f-]{36}$/);
  const profileUrl = page.url();
  const evidence = await page.getByTestId("profile-evidence").innerText();
  assert.match(evidence, /on dev: \d\.\d{3} \(/);
  assert.match(evidence, /on test: \d\.\d{3} \(/);
  await page
    .getByRole("table", { name: "Pipeline" })
    .getByRole("cell", { name: "stages.chunk.params.max_tokens" })
    .waitFor();

  await step("download it");
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Download YAML" }).click(),
  ]);
  assert.equal(download.suggestedFilename(), "kp-browser-v1.yaml");
  const saved = readFileSync(await download.path(), "utf-8");
  assert.match(saved, /^# stratafi\.knowledge-profile\.v1: kp-browser\n/);

  await step("an owner approves it");
  await switchUser(page, "owner");
  await page.goto(profileUrl);
  await page.getByRole("button", { name: "Approve" }).click();
  const decisions = page.getByRole("table", { name: "Review decisions" });
  await decisions.locator('[data-status="approved"]').waitFor();
  await decisions.getByRole("cell", { name: /owner/ }).waitFor();
  await page
    .getByRole("navigation", { name: "Stratafi" })
    .getByRole("link", { name: "Profiles", exact: true })
    .click();
  await page
    .getByRole("table", { name: "Knowledge profiles" })
    .getByRole("link", { name: "kp-browser v1" })
    .waitFor();

  await step("no query text in the console");
  for (const message of console_) {
    assert.ok(
      !/\?$/.test(message.trim()),
      "something like a query reached the console",
    );
  }
  process.stdout.write("ok\n");
} catch (error) {
  if (process.env.STRATAFI_SCREENSHOTS) {
    const [page] = browser.contexts().flatMap((context) => context.pages());
    await page?.screenshot({
      path: path.join(
        process.env.STRATAFI_SCREENSHOTS,
        "experiment-failed.png",
      ),
      fullPage: true,
    });
  }
  throw error;
} finally {
  await browser.close();
}
