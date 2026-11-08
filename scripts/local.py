#!/usr/bin/env python3
"""Run StrataFi on one machine: the model gateway, the API, the worker and the web app.

    uv run python scripts/local.py doctor   # check .env, Postgres, Ollama, the models and web/
    uv run python scripts/local.py up       # start everything; Ctrl+C stops it all
    uv run python scripts/local.py seed     # load the Halcyon demo corpus, index it, evaluate it

``up`` reads .env (``scripts/dev_setup.py`` writes it), checks what StrataFi needs, creates and
migrates the database, then starts each service as a child process with its output prefixed by
its name, and prints the lab's address once the API answers. Two things run outside it:

- **Ollama**, its own app, serving the models (``ollama pull qwen2.5:3b`` and so on).
- **Postgres 16 with pgvector**: Docker (``docker compose up -d --wait postgres``) or, on Windows
  without Docker, WSL (``scripts/wsl/postgres.sh``). With ``STRATAFI_WSL_DISTRO`` set in .env,
  ``up`` starts that distribution's Postgres and keeps WSL awake while StrataFi runs.

The settings' own defaults suit one machine: the gateway on 127.0.0.1:8091, artifacts on disk
(``data/artifacts``), one GPU slot for in-process models, and the LLM features on, so the API and
the worker need nothing more from this launcher.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
WEB = ROOT / "web"

GATEWAY_URL = "http://127.0.0.1:8091"
API_URL = "http://127.0.0.1:8000"
WEB_URL = "http://127.0.0.1:5190/stratafi/"
OLLAMA_URL = "http://127.0.0.1:11434"

# What the child processes need beyond .env: UTF-8 output on Windows consoles. .env wins.
LOCAL_DEFAULTS = {"PYTHONUTF8": "1"}

COLORS = ["\033[36m", "\033[35m", "\033[33m", "\033[32m", "\033[34m"]
RESET = "\033[0m"


# --- environment -------------------------------------------------------------------------


def read_env(path: Path) -> dict[str, str]:
    """``KEY=value`` lines, as scripts/dev_setup.py writes them (quotes optional)."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def environment() -> dict[str, str]:
    """os.environ, then .env, then the local defaults for anything still empty."""
    env = dict(os.environ)
    if ENV_FILE.exists():
        for key, value in read_env(ENV_FILE).items():
            env.setdefault(key, value)
    for key, value in LOCAL_DEFAULTS.items():
        if not env.get(key):
            env[key] = value
    return env


def tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_json(url: str, timeout: float = 3.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (local URLs)
        return json.loads(response.read().decode("utf-8"))


def database_address(env: dict[str, str]) -> tuple[str, int]:
    parsed = urllib.parse.urlsplit(env.get("STRATAFI_DATABASE_URL", ""))
    return parsed.hostname or "127.0.0.1", parsed.port or 5432


def wsl_command(distro: str, script: str) -> list[str]:
    return ["wsl.exe", "-d", distro, "-u", "root", "--", "sh", "-c", script]


def wait_for(check: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.5)
    return False


# --- doctor --------------------------------------------------------------------------------


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)

    def ok(self, message: str) -> None:
        print(f"  ok    {message}")

    def problem(self, message: str, fix: str) -> None:
        print(f"  FIX   {message}\n        {fix}")
        self.problems.append(message)


def doctor(env: dict[str, str]) -> Report:
    report = Report()
    print("Checking what StrataFi needs:")
    if ENV_FILE.exists():
        report.ok(".env exists")
    else:
        report.problem("no .env", "run: uv run python scripts/dev_setup.py")
        return report

    host, port = database_address(env)
    distro = env.get("STRATAFI_WSL_DISTRO", "")
    if tcp_open(host, port):
        report.ok(f"Postgres answers on {host}:{port}")
    else:
        fix = (
            "`scripts/local.py up` starts it, since .env names STRATAFI_WSL_DISTRO"
            if distro
            else "start it: docker compose up -d --wait postgres (or scripts/wsl/postgres.sh "
            "on Windows without Docker, then set STRATAFI_WSL_DISTRO in .env)"
        )
        report.problem(f"Postgres isn't answering on {host}:{port}", fix)

    try:
        version = http_json(f"{OLLAMA_URL}/api/version").get("version")
        report.ok(f"Ollama {version} answers on {OLLAMA_URL}")
        names = {m["name"] for m in http_json(f"{OLLAMA_URL}/api/tags").get("models", [])}
        names |= {name.removesuffix(":latest") for name in names}
        for model in ("qwen2.5:3b", "nomic-embed-text"):
            if model in names:
                report.ok(f"model {model} is pulled")
            else:
                report.problem(f"model {model} isn't pulled", f"run: ollama pull {model}")
    except (OSError, ValueError):
        report.problem(
            f"Ollama isn't answering on {OLLAMA_URL}", "start the Ollama app (ollama.com)"
        )

    if (WEB / "node_modules").is_dir():
        report.ok("web/node_modules is installed")
    else:
        report.problem("the web app's packages aren't installed", "run: cd web && npm ci")
    if shutil.which("npm") is None:
        report.problem("npm isn't on the PATH", "install Node 24 (nodejs.org)")
    return report


# --- up ------------------------------------------------------------------------------------


@dataclass
class Service:
    name: str
    argv: list[str]
    cwd: Path = ROOT
    process: subprocess.Popen[str] | None = None


def wsl_postgres(env: dict[str, str]) -> Service | None:
    """Postgres in WSL, when .env names the distribution: starting its service, then sleeping,
    which keeps WSL from stopping the distribution (and Postgres) while StrataFi runs."""
    distro = env.get("STRATAFI_WSL_DISTRO")
    if not distro or os.name != "nt":
        return None
    script = "systemctl start postgresql@16-main && echo 'postgres in WSL is up' && sleep inf"
    return Service("postgres", wsl_command(distro, script))


def services() -> list[Service]:
    python = sys.executable
    npm = (shutil.which("npm.cmd") if os.name == "nt" else None) or shutil.which("npm") or "npm"
    return [
        Service("gateway", [python, "-m", "stratafi_gateway.main"]),
        Service("api", [python, "-m", "stratafi_api.main"]),
        Service("worker", [python, "-m", "stratafi_worker.main"]),
        Service("web", [npm, "run", "dev", "--", "--host", "127.0.0.1"], cwd=WEB),
    ]


def start(service: Service, env: dict[str, str], color: str) -> None:
    service.process = subprocess.Popen(
        service.argv,
        cwd=service.cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    threading.Thread(target=pump, args=(service, color), daemon=True).start()


def pump(service: Service, color: str) -> None:
    if service.process is None or service.process.stdout is None:
        return
    prefix = f"{color}{service.name:>8}{RESET} |" if sys.stdout.isatty() else f"{service.name:>8} |"
    for line in service.process.stdout:
        print(f"{prefix} {line.rstrip()}", flush=True)


def stop(running: Sequence[Service]) -> None:
    for service in running:
        process = service.process
        if process is None or process.poll() is not None:
            continue
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                check=False,
                capture_output=True,
            )
        else:
            process.terminate()
    for service in running:
        if service.process is not None:
            try:
                service.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                service.process.kill()


def store(command: str) -> int:
    from stratafi_store.cli import main as store_main

    return store_main([command])


def up(env: dict[str, str]) -> int:
    running: list[Service] = []
    try:
        postgres = wsl_postgres(env)
        if postgres is not None:
            print(f"Starting Postgres in WSL ({env['STRATAFI_WSL_DISTRO']})…")
            start(postgres, env, COLORS[-1])
            running.append(postgres)
            wait_for(lambda: tcp_open(*database_address(env)), 60)
        report = doctor(env)
        if report.problems:
            print("\nFix the items above, then run this again.")
            return 2
        os.environ.update(env)
        print("\nPreparing the database:")
        for command in ("bootstrap", "migrate"):
            if store(command) != 0:
                print(f"stratafi-store {command} failed")
                return 1
        print("\nStarting StrataFi (Ctrl+C stops everything):")
        for index, service in enumerate(services()):
            start(service, env, COLORS[index % len(COLORS)])
            running.append(service)

        def api_live() -> bool:
            try:
                http_json(f"{API_URL}/api/v1/health/live", timeout=1)
                return True
            except (OSError, ValueError):
                return False

        if wait_for(api_live, 90):
            line = "=" * 64
            print(f"\n{line}\n  StrataFi is running: {WEB_URL}\n  API docs: {API_URL}/api/v1/docs")
            print(f"  Model gateway: {GATEWAY_URL}/v1/models\n{line}\n", flush=True)
        while True:
            for service in running:
                if service.process is not None and service.process.poll() is not None:
                    code = service.process.returncode
                    print(f"\n{service.name} stopped (exit {code}); stopping the rest.")
                    return 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping…")
        return 0
    finally:
        stop(running)


# --- seed ----------------------------------------------------------------------------------

DEMO = ROOT / "fixtures" / "demo-corpus"
CORPUS = "halcyon"
EVAL_SET = "halcyon-retrieval"
TIMEOUT = 3600.0

# A baseline for local models. Swap the embedder's alias for another Ollama embedder
# (mxbai-embed-large, all-minilm) to compare them.
PIPELINE: dict[str, Any] = {
    "schema": "stratafi.pipeline.v1",
    "name": "local-baseline",
    "description": "Markdown sections, ~400-token chunks, nomic-embed-text, exact + BM25",
    "stages": {
        "scan": {"checks": [{"component": "scan/content-safety"}, {"component": "scan/dedupe"}]},
        # The default parse route: Markdown through parse/markdown.
        "clean": [{"component": "clean/whitespace"}],
        "chunk": {"component": "chunk/sections", "params": {"max_tokens": 400, "min_tokens": 200}},
        "embed": {
            "component": "embed/gateway",
            "params": {"alias": "nomic-embed-text", "batch_size": 32},
        },
        "index": {"dense": {"component": "index/exact"}, "keyword": {"component": "index/bm25"}},
    },
}


def _retrieval(name: str, **changes: Any) -> dict[str, Any]:
    return {
        "schema": "stratafi.retrieval.v1",
        "name": name,
        "retrievers": {"dense": {"top_n": 50}, "keyword": {"top_n": 50}},
        "fusion": {"method": "rrf", "k": 60},
        "candidates": 50,
        "filters": {"current_only": True},
        "rerank": {"component": "rerank/none"},
        "top_k": 8,
        **changes,
    }


RETRIEVAL = [
    _retrieval("local-hybrid"),
    _retrieval(
        "local-dense",
        retrievers={"dense": {"top_n": 50}, "keyword": None},
        fusion={"method": "none"},
    ),
    _retrieval(
        "local-keyword",
        retrievers={"dense": None, "keyword": {"top_n": 50}},
        fusion={"method": "none"},
    ),
]


def gateway_rerankers() -> list[str]:
    """The rerank models the gateway serves (none without the rerank extra)."""
    try:
        data = http_json(f"{GATEWAY_URL}/v1/models")["data"]
    except (OSError, ValueError, KeyError):
        return []
    return [str(m["id"]) for m in data if m.get("purpose") == "rerank"]


def seed(env: dict[str, str]) -> int:
    from stratafi import ApiError, Client, StratafiError

    try:
        client = Client(API_URL, dev_sign_in=env.get("STRATAFI_DEV_SIGN_IN", "local:owner"))
        try:
            client.corpora.create(CORPUS, description="Halcyon Coffee Co.'s invented handbook")
        except ApiError as error:
            if error.status != 409:  # made by an earlier seed
                raise
        client.corpora.upload(CORPUS, manifest=DEMO / "corpus" / "manifest.yaml")
        version = client.corpora.freeze(CORPUS)
        print(f"corpus {CORPUS} v{version.version}: {version.document_count} documents")
        saved = client.pipelines.push(PIPELINE)
        print(f"pipeline {saved.name} v{saved.version}: running (the first embed takes a while)…")
        run = client.runs.start(saved, corpus=version, wait=True, timeout=TIMEOUT)
        entries = run.record.counts.entries or {}
        print(f"run {run.id}: {run.status}, {entries.get('done', 0)} documents indexed")
        if run.status != "succeeded":
            return 1
        imported = client.eval_sets.import_file(
            DEMO / "retrieval.yaml", name=EVAL_SET, test_fraction=0.3
        )
        print(f"eval set {EVAL_SET} v{imported.version.version}: {imported.report.items} items")
        configs = list(RETRIEVAL)
        if "rerank-default" in gateway_rerankers():
            rerank = {"component": "rerank/gateway", "params": {"alias": "rerank-default"}}
            configs.append(_retrieval("local-hybrid-rerank", rerank=rerank))
        ids = []
        for document in configs:
            result = client.evals.run(
                run, retrieval=document, eval_set=imported.version, wait=True, timeout=TIMEOUT
            )
            headline = result.metrics.headline() if result.status == "succeeded" else {}
            numbers = ", ".join(f"{k} {v:.3f}" for k, v in headline.items() if v is not None)
            print(f"eval {document['name']:20} {result.status}: {numbers}")
            ids.append(result.id)
        print(f"\nCompare them: {WEB_URL.rstrip('/')}/compare/{'/'.join(ids)}")
        print(f"Chat with the corpus: {WEB_URL.rstrip('/')}/chat")
        return 0
    except ApiError as error:
        print(f"seed: the API answered {error.status} {error.code or error.title}", file=sys.stderr)
        return 3
    except StratafiError as error:
        print(f"seed: {error}", file=sys.stderr)
        return 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run StrataFi on one machine.")
    parser.add_argument("command", choices=["doctor", "up", "seed"])
    args = parser.parse_args(argv)
    if os.name == "nt":
        os.system("")  # noqa: S605 (turns on ANSI colors in older Windows consoles)
    env = environment()
    if args.command == "doctor":
        return 1 if doctor(env).problems else 0
    if args.command == "up":
        return up(env)
    return seed(env)


if __name__ == "__main__":
    sys.exit(main())
