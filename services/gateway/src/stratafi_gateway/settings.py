"""The gateway's settings, read from the environment (``GatewaySettings.from_env``).

Every variable starts with ``STRATAFI_GATEWAY_``:

- ``OLLAMA_URL``: where Ollama listens (``http://127.0.0.1:11434``)
- ``ALIASES``: more names for Ollama models, ``alias=model,...``, merged over
  ``DEFAULT_ALIASES``
- ``RERANKERS``: the cross-encoders on ``/v1/rerank``, ``alias=hf-id,...``, replacing
  ``DEFAULT_RERANKERS``
- ``NUM_CTX``: the context window Ollama allocates for chat (8192)
- ``KEEP_ALIVE``: how long Ollama keeps a model loaded, e.g. ``30m`` (Ollama's own default)
- ``TRUNCATE``: truncate embedding inputs longer than the model's context (``true``)
- ``RERANK_DEVICE``: ``cpu`` or ``cuda`` for the cross-encoders (automatic)
- ``TIMEOUT_SECONDS``: how long one call to Ollama may take; a cold model loads first (600)
- ``HOST`` and ``PORT``: where the gateway listens (``127.0.0.1`` and ``8091``)

Every Ollama model is also served under its own name (``qwen2.5:3b``, ``nomic-embed-text``), so
an alias is only a convenience: pipeline specs can name models directly.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

ALIAS_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")

# The aliases StrataFi's own specs and examples use; change them to try other models without
# editing a spec. Each one is listed only while the model it names is pulled.
DEFAULT_ALIASES: dict[str, str] = {
    "embed-default": "nomic-embed-text",
    "chat-default": "qwen2.5:3b",
}
DEFAULT_RERANKERS: dict[str, str] = {
    "rerank-default": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


def parse_pairs(text: str, *, name: str) -> dict[str, str]:
    """``"a=b, c=d"`` as ``{"a": "b", "c": "d"}``; every key must be a valid alias."""
    pairs: dict[str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        alias, sep, target = (part.strip() for part in item.partition("="))
        if not sep or not target or ALIAS_RE.fullmatch(alias) is None:
            raise ValueError(f"{name}: {item!r} is not alias=target")
        pairs[alias] = target
    return pairs


def _flag(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class GatewaySettings:
    ollama_url: str = "http://127.0.0.1:11434"
    aliases: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    rerankers: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_RERANKERS))
    num_ctx: int = 8192
    keep_alive: str | None = None
    truncate: bool = True
    rerank_device: str | None = None
    timeout_seconds: float = 600.0
    models_ttl_seconds: float = 10.0
    host: str = "127.0.0.1"
    port: int = 8091

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "GatewaySettings":
        env = os.environ if environ is None else environ

        def get(key: str) -> str | None:
            value = env.get(f"STRATAFI_GATEWAY_{key}")
            return value if value not in (None, "") else None

        aliases = dict(DEFAULT_ALIASES)
        if (value := get("ALIASES")) is not None:
            aliases.update(parse_pairs(value, name="STRATAFI_GATEWAY_ALIASES"))
        rerankers = dict(DEFAULT_RERANKERS)
        if (value := get("RERANKERS")) is not None:
            rerankers = parse_pairs(value, name="STRATAFI_GATEWAY_RERANKERS")
        return cls(
            ollama_url=get("OLLAMA_URL") or cls.ollama_url,
            aliases=aliases,
            rerankers=rerankers,
            num_ctx=int(get("NUM_CTX") or cls.num_ctx),
            keep_alive=get("KEEP_ALIVE"),
            truncate=_flag(get("TRUNCATE") or "true"),
            rerank_device=get("RERANK_DEVICE"),
            timeout_seconds=float(get("TIMEOUT_SECONDS") or cls.timeout_seconds),
            host=get("HOST") or cls.host,
            port=int(get("PORT") or cls.port),
        )
