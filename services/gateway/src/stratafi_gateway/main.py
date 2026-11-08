"""``stratafi-gateway``: serve the local model gateway (settings from the environment)."""

import logging

import uvicorn

from stratafi_gateway.app import create_app
from stratafi_gateway.settings import GatewaySettings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # one line per Ollama call is noise
    settings = GatewaySettings.from_env()
    app = create_app(settings)
    rerankers = ", ".join(app.state.rerankers.aliases) or "none (install the rerank extra)"
    logging.getLogger("stratafi_gateway").info(
        "serving on http://%s:%d, Ollama at %s, aliases %s, rerankers %s",
        settings.host,
        settings.port,
        settings.ollama_url,
        ", ".join(f"{a}={m}" for a, m in settings.aliases.items()),
        rerankers,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
