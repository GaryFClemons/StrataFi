# stratafi-core

StrataFi's domain package. Every other package depends on it, so it holds no service code:

- the pipeline spec and retrieval config (`specs.py`), eval sets, experiments, test suites and
  knowledge profiles, with their validation
- canonical JSON and cache keys (`hashing.py`), artifact and run manifest models
- the component interface and registry (`components.py`)
- relevance rules, metrics and statistics (`relevance.py`, `metrics.py`, `stats.py`)
- the in-process PII detector (`pii.py`) and the chat prompt (`chat.py`)
- tracking: annotations, the search language, traces, model comparison and pipeline maps
- settings (`settings.py`), identity tokens (`identity.py`) and metadata-only logging (`logs.py`)

[The architecture](../../docs/architecture.md) explains how these fit together.
