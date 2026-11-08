# stratafi-gateway

StrataFi's local model gateway. It serves the model gateway contract that every model component
uses (`embed/gateway`, `rerank/gateway`, `generate/gateway`, `rerank/llm-listwise`,
`enrich/llm-context`) on one machine:

- **Embeddings and chat** come from [Ollama](https://ollama.com). Every model you've pulled is
  served under its own name (`qwen2.5:3b`, `nomic-embed-text`), plus a few aliases
  (`embed-default`, `chat-default`).
- **Reranking** comes from Hugging Face cross-encoders through sentence-transformers (the
  `rerank` extra: `uv sync --all-packages --locked --extra rerank`), since Ollama serves no
  rerankers. Without the extra, `/v1/models` lists no rerankers and `/v1/rerank` answers 404.
- **Revisions:** every response names the model's revision in `X-Model-Revision`: the weights'
  digest for Ollama models. Pulling a model again changes it, so cached embeddings are never
  mixed across weights.
- **Speed:** chat responses report load, prompt and completion times and tokens per second, which
  the Chat screen shows with each answer.

```bash
ollama pull qwen2.5:3b && ollama pull nomic-embed-text
uv run stratafi-gateway            # http://127.0.0.1:8091
curl http://127.0.0.1:8091/v1/models
```

Settings are environment variables (`STRATAFI_GATEWAY_*`), documented in
[`settings.py`](src/stratafi_gateway/settings.py). For example, to make the default chat model
Llama and add a second reranker:

```bash
STRATAFI_GATEWAY_ALIASES="chat-default=llama3.2:3b" \
STRATAFI_GATEWAY_RERANKERS="rerank-default=cross-encoder/ms-marco-MiniLM-L-6-v2,bge-reranker-base=BAAI/bge-reranker-base" \
uv run stratafi-gateway
```

It takes no credentials and listens on 127.0.0.1: it's for your own machine. It never logs
request or response text.
