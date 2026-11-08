// Shared steps for the step-screen browser flows (S3.x): launch Chromium,
// sign in, make a corpus from files and freeze it, run a pipeline step, and
// check that no document text reached the console.

import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

export const RUN_TIMEOUT = 120_000;

export async function open() {
  const browser = await chromium.launch(
    process.env.STRATAFI_CHROMIUM
      ? { executablePath: process.env.STRATAFI_CHROMIUM }
      : {},
  );
  const page = await browser.newPage();
  page.setDefaultTimeout(30_000);
  const console_ = [];
  page.on("console", (message) => console_.push(message.text()));
  page.on("pageerror", (error) => console_.push(String(error)));
  let shots = 0;
  const step = async (name) => {
    if (process.env.STRATAFI_SCREENSHOTS && shots > 0) {
      await page.screenshot({
        path: path.join(process.env.STRATAFI_SCREENSHOTS, `${shots}.png`),
        fullPage: true,
      });
    }
    shots += 1;
    process.stdout.write(`- ${name}\n`);
  };
  return { browser, page, console_, step };
}

/**
 * Open the lab, which signs in as the local owner by itself, and switch to
 * the synthetic user of the tenant whose data the flow uses.
 */
export async function signIn(
  page,
  appUrl,
  role = "editor",
  tenant = "tenant0",
) {
  await page.goto(appUrl);
  await page.getByRole("navigation", { name: "Stratafi" }).waitFor();
  await switchUser(page, role, tenant);
}

/** Switch user from the frame: another tenant, another role. */
export async function switchUser(page, role, tenant = "tenant0") {
  await page.getByRole("button", { name: "Switch user" }).click();
  await page.getByLabel("Tenant").fill(tenant);
  await page.getByLabel("Role").selectOption(role);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.getByRole("navigation", { name: "Stratafi" }).waitFor();
}

/** Create a corpus, upload files into it, and freeze version 1. */
export async function corpusFrom(page, name, files, expected) {
  await page.getByRole("link", { name: "Documents", exact: true }).click();
  await page.getByLabel("New corpus").fill(name);
  await page.getByRole("button", { name: "Create corpus" }).click();
  await page.getByText(`Corpus ${name}`).waitFor();
  await page.getByLabel("Upload documents").setInputFiles(files);
  await page.getByRole("button", { name: "Upload", exact: true }).click();
  const documents = page.getByRole("table", { name: "Documents" });
  for (const entry of expected) {
    await documents.getByRole("cell", { name: entry, exact: true }).waitFor();
  }
  await page.getByRole("button", { name: "Freeze version" }).click();
  await page
    .getByRole("table", { name: "Versions" })
    .getByRole("cell", { name: "v1", exact: true })
    .waitFor();
}

export async function waitForRun(page, status = "succeeded") {
  await page
    .locator(`[data-testid="run-progress"] [data-status="${status}"]`)
    .last()
    .waitFor({ timeout: RUN_TIMEOUT });
}

/** Open a pipeline step and run to it on every document of a corpus's v1. */
export async function runStep(page, label, corpusLabel) {
  await page.getByRole("link", { name: "Pipelines", exact: true }).click();
  await page.getByRole("link", { name: label }).click();
  await page.getByLabel("Corpus version").selectOption({ label: corpusLabel });
  await page.getByLabel("Sample").selectOption("all");
  await page.getByRole("button", { name: "Run to this step" }).click();
  await waitForRun(page);
}

export function assertNoText(console_, phrases) {
  for (const message of console_) {
    for (const phrase of phrases) {
      assert.ok(!message.includes(phrase), "document text reached the console");
    }
  }
}
