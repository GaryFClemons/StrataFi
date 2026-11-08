# Halcyon Coffee demo corpus

The internal knowledge base of **Halcyon Coffee Co.**, an invented specialty coffee roaster with
40 cafés, the Fenwick Street roastery and a wholesale business, plus a retrieval eval set of 88
staff questions about it. It's StrataFi's demo set: small enough to read in an afternoon, with
traps that separate good retrieval setups from bad ones. These include superseded and
not-yet-effective versions, documents only some people may read, answers that sit only in a
table row, and questions no document answers.

Everything here is invented: the company, its people, places, numbers and rules. Any resemblance
to a real business is a coincidence. The folder is released under
[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/), so you can copy, change and
reuse it without asking.

## Layout

| Path | What it is |
| --- | --- |
| [`corpus/`](corpus/) | 24 Markdown documents (11,546 words), with numbered section headings |
| [`corpus/manifest.yaml`](corpus/manifest.yaml) | Each document's path, collection, key, version, effective dates, title, ACL, source URI and status, as the corpus manifest that `extract/manifest` imports |
| [`retrieval.yaml`](retrieval.yaml) | The eval set: 88 queries with their asker, gold sections and evidence (`stratafi.eval-set.v1`'s retrieval format) |
| [`check_eval_set.py`](check_eval_set.py) | Checks that the eval set and the corpus agree |

## The corpus

| Collection | Readable by (ACL groups) | Documents |
| --- | --- | --- |
| `handbook` | `all-staff` | Paid Time Off and Leave Policy (v2 superseded, v3 current); Scheduling and Shift Swap Policy; Tip Policy; Dress Code and Appearance Standard; Workplace Safety and Injury Reporting; Code of Conduct; Barista Training and Certification Program; HQ Remote Work and Expense Policy; Policy Index and Owners |
| `operations` | `all-staff` | Food Safety and Allergen Standard; Espresso and Brew Standards; Equipment Cleaning Schedule; Opening and Closing Procedures; Refunds, Remakes and Guest Recovery; Halcyon Rewards Program (v2 superseded, v3 current, v4 from 2027-01-01); Mobile Order and Pay Procedures; Wholesale Pricing and Ordering Guide |
| `management` | `managers` | Cash Handling and Safe Procedures; Hiring and Pay Bands; Performance and Discipline Guide |
| `executive` | `executives` | 2027 Expansion Plan (a confidential memo) |

Each document opens with a version line (version, effective date, owner, approval, the version it
replaces) and uses numbered headings (`## 3 Shift swaps`, `### 3.1 Trading a shift with a
coworker`). An eval item's `section` is that heading's text, which is also the section label the
default chunker (`chunk/sections`) gives the document's chunks. The facts are kept consistent across documents.
For example, the refund limits, the mobile-order refund rule, the opening checklist and the
cleaning schedule all agree with each other.

## The traps

The eval set's "today" (`as_of`) is **2026-11-01**.

- **Superseded versions.** Paid Time Off and Leave v2 was replaced by v3 on 2026-01-01, and
  Halcyon Rewards v2 by v3 on 2025-09-01. Both old versions are still in the corpus, with
  different numbers for the same rules (accrual, bereavement days, notice, parental leave,
  payout; stars instead of points). Version questions list them in `avoid`: they must not
  outrank the current version.
- **A future-dated version.** Halcyon Rewards v4 is approved but takes effect only on
  2027-01-01, so v3 is current. v4 changes the free-drink threshold (60 to 75 points), point
  expiry (12 to 9 months), the birthday reward, the Gold threshold (500 to 600) and the
  adjustment limit, and adds a reusable-cup bonus. Questions about those rules expect v3 and
  avoid v4. One question (`version-reusable-cup-bonus`) is answered only by v4, so as of
  2026-11-01 it expects nothing and must not return v4.
- **Access control.** `management/` is readable only by `managers` and `executive/` only by
  `executives`. Each restricted question is asked by an asker who may read the document, and
  again by one who may not (IDs ending `--as-staff` or `--as-manager`). That second copy expects
  nothing and lists the document in `must_not_return`. `table-safe-procedure-owner` asks staff
  who owns the cash procedure, which only the all-staff Policy Index can answer.
- **Tables.** Some answers sit only in a table row: holding times, espresso and brew recipes,
  the cleaning schedule, wholesale price tiers, pay bands, attendance points and policy owners.
  Their evidence phrases are verbatim runs of cells, for example
  `Decaf espresso, double | 18 g | 38 g | 28–33 seconds`.
- **Abstention.** Thirteen plausible questions have no answer in any current document (guest
  Wi-Fi, payday, a retirement match, the oat milk upcharge, and so on). Each lists its
  `near_misses`, the closest passages that a good answer cites while saying the policy doesn't
  cover the question. `abstain-partial-curbside` is half answerable: the shelf time is covered,
  curbside service isn't.

## The eval set

`retrieval.yaml` has `tenant: halcyon`, `as_of: '2026-11-01'`, `k: 5`, its own `askers` table and
88 `queries`. The askers are what document ACLs are checked against:

| Asker | Groups | Items |
| --- | --- | --- |
| `staff` | `all-staff` | 76 |
| `manager` | `all-staff`, `managers` | 9 |
| `exec` | `all-staff`, `managers`, `executives` | 3 |

Each item's ID starts with its type:

| Type | Items | What it tests |
| --- | --- | --- |
| `lookup-` | 22 | One fact, mostly asked in other words than the document's ("airpot" for batch brew server, "non-slip shoes" for slip-resistant ones) |
| `scenario-` | 12 | A situation that needs two or three sections, sometimes from different documents |
| `table-` | 12 | Facts that appear only in a table row |
| `version-` | 13 | Rules that changed between versions: 7 against PTO v2, 6 against Rewards v2 or the future-dated v4 |
| `acl-` | 16 | 9 questions for managers or executives, and 7 copies asked by someone who may not read the answer |
| `abstain-` | 13 | No current document answers (11 asked by staff, 1 by a manager, 1 by an executive) |
| **Total** | **88** | 81 expected evidence items and 30 near misses |

A result is a hit when it comes from an expected `doc_key` and version and either its section
matches or its text contains the evidence phrase. Every evidence phrase is a verbatim substring,
at most 100 characters long, of its section's own text. Items with `avoid` (13) must not rank an
avoided version above the first hit. Items with `must_not_return` (9) must never return that
document for that asker on that day. Items with `expected: []` (21) have no hit to find.

The default chunker (`chunk/sections`, 200 to 400 tokens) starts a chunk at each top-level section
and packs smaller subsections together, so sections compete with each other, which is where
paraphrased lookups and tables get harder. Much larger chunks put most of these short documents
into one or two chunks, and retrieval becomes close to picking the right document.

## Using it

Check the set after any edit, from the repository root:

```bash
uv run --no-sync python fixtures/demo-corpus/check_eval_set.py
```

It loads `retrieval.yaml` with `stratafi_core.evalsets.load_eval_set` and resolves it with
`resolve_eval_set`, loads the manifest with `load_corpus_manifest`, and then checks that every
reference names a document and version in the manifest, every `section` is a heading of that
document, and every `evidence` phrase is verbatim in that section. It also checks that every
asker can read what it expects, that expected versions are in effect on `as_of` while avoided
ones aren't, that every `must_not_return` is out of the asker's sight, and that each type follows
its rules. It prints the counts by type and exits with 1 if anything is wrong.

To load it into a lab, import the corpus with its metadata (`extract/manifest`), then the eval
set:

- API: `POST /api/v1/documents/import-manifest` with root `fixtures` and path
  `demo-corpus/corpus`, then `POST /api/v1/eval-sets/import` with `retrieval.yaml` (format
  `yaml`).
- SDK: `client.corpora.upload(corpus, manifest="fixtures/demo-corpus/corpus/manifest.yaml")`,
  then `client.eval_sets.import_file("fixtures/demo-corpus/retrieval.yaml", name="halcyon-retrieval")`.

Search with the retrieval config's `current_only` filter on, so `as_of` hides superseded and
future-dated versions. Turn it off to see the version traps bite.

## Editing

- Keep evidence phrases verbatim. If you reword a sentence that an item quotes, update the item
  too, and rerun the checker.
- New versions go in as new files (`<key>-v<N>.md`) with a manifest entry. Give the version it
  replaces an `effective_to` equal to the new version's `effective_from`.
- Keep the four collections' ACLs as they are, or update the askers and the `acl-` items with
  them.
