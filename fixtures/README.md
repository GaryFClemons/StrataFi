# Fixtures

Synthetic documents for tests and local development. Everything here is invented: the lab uses
synthetic data only. The demo corpus is released under CC0-1.0 ([its README](demo-corpus/README.md)),
and `formats/` and `multi-format/` are rendered from its documents.

## `demo-corpus/`

The Halcyon Coffee demo corpus: 24 invented Markdown documents with a manifest, and a retrieval
eval set of 88 staff questions with its own askers, `as_of` 2026-11-01. Its
[README](demo-corpus/README.md) describes the corpus, its traps and the checker. It's the corpus
for the lab's demo, the retrieval and relevance tests and the integration tier
(`stratafi_testing.lab.prepare_halcyon`).

- `POST /api/v1/eval-sets/import` takes `retrieval.yaml` unchanged (`stratafi.eval-set.v1` reads
  its retrieval format).
- `POST /api/v1/documents/import-manifest` (root `fixtures`, path `demo-corpus/corpus`) imports
  the corpus with its metadata (`extract/manifest`).

## `formats/`

Three of the demo corpus's documents (the Tip Policy, the Wholesale Pricing and Ordering Guide
and the Food Safety and Allergen Standard), each as Markdown, plain text, DOCX and PDF, for the
parsers' tests. `scripts/format_fixtures.py` writes them byte for byte; `--check` proves it still
does. The plain-text files use Setext headings, wrapped lines, CRLF and a byte-order mark; the
DOCX files are stored unzipped and the PDFs uncompressed, so their bytes don't depend on zlib.

```bash
uv run python scripts/format_fixtures.py
```

Rewrite the parsers' goldens after writing them again (`scripts/parser_goldens.py`, below). The
same files are a small corpus for end-to-end runs and the UI's browser flows; in development
the API can import them with `POST /api/v1/documents/import-folder` (root `fixtures`).

## `multi-format/`

Two of the demo corpus's documents (`corpus/`, with a manifest in its format) and five questions
about them (`retrieval.yaml`; two have their evidence in tables), rendered by
`scripts/corpus_variants.py` as every variant in `variants/`: DOCX, digital PDF, scanned
PDF (image-only pages at 150 dpi, seed 0), HTML, and PDF with tables as images. Each variant's
`manifest.yaml` keeps the documents' keys and metadata, and its `variant.json` records the seed,
the source manifest's SHA-256 and the libraries' versions. Render them again after changing the
corpus or the script:

```bash
uv run python scripts/corpus_variants.py fixtures/multi-format/corpus fixtures/multi-format/variants
```

The demo corpus's own variants are generated rather than committed. `--evidence` reports how
much of an eval set's evidence each variant keeps (all of it in DOCX, PDF and HTML; the
tables-as-images PDF loses the phrases only its tables hold), and `--import-as` imports each
variant as a corpus through the API (the SDK's settings):

```bash
uv run python scripts/corpus_variants.py fixtures/demo-corpus/corpus data/variants/halcyon --evidence fixtures/demo-corpus/retrieval.yaml
```

## `parsers/goldens/`

What each parser makes of the format fixtures it accepts: every element's type, level, page,
whether it has a box, and its text. `scripts/parser_goldens.py` writes them for the parsers this
machine can run, so a parser whose extra isn't installed keeps its old goldens; the unit tier
compares (`packages/components/tests/test_parsers.py`). A parser whose output changes bumps its
version and rewrites its goldens.

```bash
uv run python scripts/parser_goldens.py
```
