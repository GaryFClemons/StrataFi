// The post-training browser flow: mine a training set from a dev eval
// run, train a query adapter on it, follow the run to its epochs, open the
// registered model's card, and evaluate the adapter on the dev split, which
// lands on the comparison with the base. Nothing the training set quotes may
// reach the browser console.
//
//   node e2e/training-flow.mjs <app URL> <a query the eval set holds>
//
// services/worker/tests/test_training_integration.py runs the Halcyon demo corpus
// through a pipeline run and a dev eval run first, serves the built app
// beside the API, and runs this. STRATAFI_SCREENSHOTS=<dir> saves a
// screenshot after each step.

import assert from "node:assert/strict";
import { open, signIn } from "./lib.mjs";

const [appUrl, query] = process.argv.slice(2);
if (!appUrl || !query) {
  console.error("usage: node e2e/training-flow.mjs <app URL> <a query>");
  process.exit(2);
}

const TRAINING_TIMEOUT = 180_000;
const { browser, page, console_, step } = await open();
try {
  await step("sign in as an editor");
  await signIn(page, appUrl);

  await step("mine a training set from the dev eval run");
  await page
    .getByRole("navigation", { name: "Stratafi" })
    .getByRole("link", { name: "Training", exact: true })
    .click();
  await page.getByRole("button", { name: "Mine", exact: true }).click();
  await page.getByText(/^Training set /).waitFor();
  await page
    .locator('[data-status="succeeded"]')
    .first()
    .waitFor({ timeout: TRAINING_TIMEOUT });
  await page
    .getByRole("term")
    .filter({ hasText: /^Hard negatives$/ })
    .waitFor();
  await page.getByRole("heading", { name: "Records" }).waitFor();

  await step("train a query adapter on it");
  const train = page.getByRole("button", { name: "Train", exact: true });
  await train.waitFor();
  await page.getByLabel("Model name").fill("halcyon-adapter-browser");
  await train.click();
  await page.getByText(/^Training run: halcyon-adapter-browser/).waitFor();
  await page
    .locator('[data-status="succeeded"]')
    .first()
    .waitFor({ timeout: TRAINING_TIMEOUT });

  await step("its epochs are charted");
  await page
    .locator('[data-testid="training-epochs"][data-rendered="true"]')
    .waitFor();
  await page.getByText(/^Kept epoch /).waitFor();

  await step("open the registered model and its card");
  await page
    .getByRole("link", { name: "halcyon-adapter-browser", exact: true })
    .click();
  await page.getByText("halcyon-adapter-browser@1").waitFor();
  await page
    .getByText(/retrievers\.dense\.adapter/)
    .first()
    .waitFor();
  await page.getByRole("heading", { name: "Limits" }).waitFor();

  await step("evaluate it on the dev split, against the base");
  await page.getByRole("button", { name: "Evaluate on dev" }).click();
  await page.waitForURL(/\/evals\/compare\/[0-9a-f-]+\/[0-9a-f-]+$/, {
    timeout: 60_000,
  });

  await step("the model is listed on the Training screen");
  await page
    .getByRole("navigation", { name: "Stratafi" })
    .getByRole("link", { name: "Training", exact: true })
    .click();
  const models = page.getByRole("table", { name: "Trained models" });
  await models
    .getByRole("link", { name: "halcyon-adapter-browser@1" })
    .waitFor();

  const logged = console_.join("\n");
  assert.ok(!logged.includes(query), "a query reached the browser console");
  assert.ok(
    !console_.some((line) => /Uncaught|TypeError|ReferenceError/.test(line)),
    `the page threw: ${logged}`,
  );
  await step("done");
  console.log("ok");
} finally {
  await browser.close();
}
