"""``stratafi-api``: run the API with JSON logs.

    uv run --env-file .env stratafi-api

Settings come from ``config/stratafi.yaml`` and the environment (``stratafi_core.settings``).
Uvicorn runs without its access log, which would record raw paths and query strings; the request
middleware writes one metadata-only line per request instead.
"""

import sys

import uvicorn

from stratafi_api.app import create_app
from stratafi_components.confine import harden_process
from stratafi_core.logs import configure_logging
from stratafi_core.settings import Settings


def main() -> int:
    settings = Settings()
    configure_logging("stratafi-api", level=settings.log_level)
    # A prod configuration must confine parsers and guard its metrics.
    problems = settings.deployment_problems()
    for problem in problems:
        print(f"stratafi-api: {problem}", file=sys.stderr)
    if problems:
        return 2
    # No process of the same user reads this one's memory or environment.
    harden_process()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_config=None,
        access_log=False,
        server_header=False,
        proxy_headers=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
