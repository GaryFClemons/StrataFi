# stratafi-components

StrataFi's built-in pipeline components, one class per implementation of a stage, and what
they need to run safely: the parse sandbox and its confinement (`sandbox.py`, `confine.py`),
the URL guard (`fetch.py`), archive limits (`archives.py`) and the model gateway client
(`models.py`). `default_registry()` holds all 67:

| Stage | Components |
| --- | --- |
| Extract | `extract/upload`, `folder`, `manifest`, `s3`, `url`; `unpack/zip`, `unpack/email` |
| Scan | `scan/content-safety`, `clamav`, `pii`, `dedupe` |
| Parse | `parse/markdown`, `pypdf`, `pdfplumber`, `markitdown`, `docling`, `unstructured`, `spreadsheet`, `pptx`, `html`, `tika`, `ocr`, `pymupdf4llm`, and hosted adapters (`azure-document-intelligence`, `aws-textract`, `google-document-ai`, `llamaparse`, `mistral-ocr`) |
| Clean | `clean/whitespace`, `unicode`, `dehyphenate`, `redact`, `repeated-lines`, `boilerplate`, `near-duplicates`, `language`, `min-content` |
| Chunk | `chunk/sections`, `fixed-tokens`, `recursive`, `sentence-window`, `semantic`, `table-aware`, `parent-child` |
| Enrich | `enrich/none`, `enrich/llm-context` |
| Embed | `embed/gateway`, `sentence-transformers`, `openai-compatible`, `fake-hash` |
| Index | `index/exact`, `bm25`, `postgres-fts`, `pgvector-hnsw` |
| Rerank | `rerank/none`, `gateway`, `cross-encoder`, `llm-listwise`, `cohere`, `jina`, `voyage`, `fake-overlap` |
| Generate | `generate/gateway`, `generate/fake` |
| Train | `train/query-adapter`, `sentence-transformers`, `cross-encoder` |

Each component declares its ID, version, parameter model, the types it accepts, its resource
class, license, platforms, egress and the services it uses
([the architecture](../../docs/architecture.md#pipelines-and-specs)).

- **Extras.** Heavy libraries sit behind extras of this package: `pdfplumber`, `ocr`, `docling`
  and `unstructured` (Linux only), `sentence-transformers`, `umap`, `hf`, `textract` and
  `pymupdf4llm`. A component whose extra isn't installed stays in the registry, says which
  extra it needs, and refuses to run (`extra_missing`).
- **Licenses.** AGPL and other licenses that need review are optional extras only, and their
  components refuse to run until `licenses.allowed` names the license.
- **Versions.** A component bumps its `version` whenever its output could change, because the
  version is part of every cache key. Parsers are held to goldens on the format fixtures
  (`scripts/parser_goldens.py`, `tests/test_parsers.py`).
