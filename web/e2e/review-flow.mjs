// The question review browser flow: draft questions from a run's passages, review
// them in the queue (edit and accept one, accept one, reject one), add the
// accepted ones to a new eval set version; then answer an eval run's queries,
// read the metrics and label an answer for calibration. No question text may
// reach the browser console.
//
//   node e2e/review-flow.mjs <app URL> <eval set name> <eval run ID>
//
// services/worker/tests/test_answers_integration.py runs the Halcyon demo corpus
// through a pipeline run and an eval run first, serves the built app beside
// the API, and runs this. STRATAFI_SCREENSHOTS=<dir> saves a screenshot after
// each step.

import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";
import { signIn } from "./lib.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [appUrl, evalSetName, evalRunId] = process.argv.slice(2);
if (!appUrl || !evalSetName || !evalRunId) {
  console.error(
    "usage: node e2e/review-flow.mjs <app URL> <eval set name> <eval run ID>",
  );
  process.exit(2);
}

const EDITED = "How many days is an account new after it is opened?";

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

  let shots = 0;
  const step = async (name) => {
    if (process.env.STRATAFI_SCREENSHOTS && shots > 0) {
      await page.screenshot({
        path: path.join(
          process.env.STRATAFI_SCREENSHOTS,
          `review-${shots}.png`,
        ),
        fullPage: true,
      });
    }
    shots += 1;
    process.stdout.write(`- ${name}\n`);
  };

  await step("sign in as an editor");
  await signIn(page, appUrl);
  const navigation = page.getByRole("navigation", { name: "Stratafi" });

  await step("open the eval set");
  await navigation
    .getByRole("link", { name: "Eval sets", exact: true })
    .click();
  await page.getByRole("link", { name: evalSetName, exact: true }).click();
  const versions = page.getByRole("table", { name: "Versions" });
  await versions.waitFor();
  const before = await versions.locator("tbody tr").count();

  await step("draft four questions from the run's passages");
  const form = page.getByRole("form", { name: "Draft questions" });
  const runs = form.getByLabel("From run");
  await runs.locator("option").nth(1).waitFor({ state: "attached" });
  await runs.selectOption({ index: 1 });
  await form.getByLabel("Generator").selectOption("generate/fake");
  await form.getByLabel("Questions").fill("4");
  await form.getByRole("button", { name: "Draft questions" }).click();
  const counts = page.getByTestId("queue-counts");
  await page.waitForFunction(() =>
    document
      .querySelector('[data-testid="queue-counts"]')
      ?.textContent?.includes("4 proposed"),
  );
  const queue = page.getByRole("table", { name: "Drafted questions" });
  assert.equal(await queue.getByRole("row").count(), 5); // a header and four

  await step("edit and accept one, accept one, reject one");
  const row = (index) => queue.getByRole("row").nth(index);
  await row(1).getByRole("button", { name: "Edit" }).click();
  await row(1).getByLabel("Edited question").fill(EDITED);
  await row(1).getByLabel("Split").selectOption("test");
  await row(1).getByRole("button", { name: "Save and accept" }).click();
  await page.waitForFunction(() =>
    document
      .querySelector('[data-testid="queue-counts"]')
      ?.textContent?.includes("3 proposed"),
  );
  // Accepted questions leave the "proposed" view, so the next one is first.
  await row(1).getByRole("button", { name: "Accept" }).click();
  await page.waitForFunction(() =>
    document
      .querySelector('[data-testid="queue-counts"]')
      ?.textContent?.includes("2 proposed"),
  );
  await row(1).getByRole("button", { name: "Reject" }).click();
  await page.waitForFunction(() =>
    document
      .querySelector('[data-testid="queue-counts"]')
      ?.textContent?.includes("1 proposed"),
  );
  assert.match((await counts.textContent()) ?? "", /2 accepted/);
  assert.match((await counts.textContent()) ?? "", /1 rejected/);

  await step("add the accepted questions to a new version");
  await page
    .getByRole("button", { name: "Add 2 accepted to a new version" })
    .click();
  await page.waitForFunction(
    (rows) =>
      document.querySelectorAll('table[aria-label="Versions"] tbody tr')
        .length === rows,
    before + 1,
  );
  await page.getByLabel("Show").selectOption("added");
  await queue.getByRole("row").nth(2).waitFor();
  assert.equal(await queue.getByRole("row").count(), 3);
  assert.equal(await queue.getByText(EDITED).count(), 1);
  assert.equal(await queue.getByText(/^gen-[0-9a-f]{12}$/).count(), 2);
  const items = page.getByRole("table", { name: "Items" });
  await items.getByText(EDITED).waitFor();

  await step("answer an eval run's queries");
  await navigation.getByRole("link", { name: "Evals", exact: true }).click();
  await page
    .getByRole("table", { name: "Eval runs" })
    .getByRole("link", { name: evalRunId.slice(0, 8) })
    .click();
  const answering = page.getByRole("form", { name: "Answer the queries" });
  await answering.getByLabel("Generator").selectOption("generate/fake");
  await answering.getByLabel("Judge").selectOption("generate/fake");
  await answering.getByRole("button", { name: "Answer the queries" }).click();
  const answerRuns = page.getByRole("table", { name: "Answer runs" });
  await answerRuns.locator('[data-status="succeeded"]').waitFor();
  await answerRuns.getByRole("link").first().click();

  await step("read the metrics and label an answer");
  const metrics = page.getByRole("table", { name: "Answer metrics" });
  await metrics.waitFor();
  assert.match(
    (await metrics.getByRole("row").nth(1).textContent()) ?? "",
    /Abstention accuracy\d\.\d{3} \(\d\.\d{3}–\d\.\d{3}\)/,
  );
  const answers = page.getByRole("table", { name: "Answers" });
  await answers.getByRole("row").nth(1).waitFor();
  await answers
    .getByRole("button", { name: /^Faithful: yes for / })
    .first()
    .click();
  const agreement = page.getByRole("table", { name: "Agreement with labels" });
  await page.waitForFunction(() =>
    /^Faithful1/.test(
      document.querySelector(
        'table[aria-label="Agreement with labels"] tbody tr',
      )?.textContent ?? "",
    ),
  );
  assert.match(
    (await agreement.getByRole("row").nth(1).textContent()) ?? "",
    /^Faithful1/,
  );
  await answers.getByRole("row").nth(1).click();
  await page.getByRole("table", { name: "Passages given" }).waitFor();

  await step("no question text in the console");
  for (const line of console_) {
    assert.ok(!line.includes(EDITED), "a question reached the console");
  }
  process.stdout.write("ok\n");
} finally {
  await browser.close();
}
