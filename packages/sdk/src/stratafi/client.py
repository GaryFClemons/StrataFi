"""The Stratafi client.

    from stratafi import Client

    client = Client.from_env()
    run = client.runs.start("baseline-v1", corpus="halcyon@latest", wait=True)

``Client.from_env`` reads:

- ``STRATAFI_URL``: the API's address, such as ``http://localhost:8100`` (``/api/v1`` is added).
- ``STRATAFI_TOKEN``: an API token (``sft_...``), sent as ``Authorization: Bearer``; or
  an identity token, sent as ``X-Stratafi-Identity``.
- ``STRATAFI_DEV_SIGN_IN``: instead of a token, ``tenant:role`` (``tenant0:editor``); with
  neither set, ``local:owner``, the lab's one local person. The client asks the API's
  development sign-in route for a short-lived token and asks again as it ages, so a
  long wait doesn't outlive its token.

Every resource answers with the API's own models (``stratafi.models``, generated from the
OpenAPI contract), wrapped where a wrapper helps: ``Run`` and ``EvalRun`` wait, ``Pipeline``
makes variants, ``Records``, ``Metrics`` and ``Comparison`` convert to dicts and DataFrames.

Waiting follows the run's server-sent events (``GET /events?run_id=``). Events are hints, as the
API says, so the record is read again after each connection, when an event says the run
finished, when the stream lags, and every ``RECHECK_SECONDS``; a broken stream reconnects with
exponential backoff, and an API without events is polled.
"""

import json
import os
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx2

from stratafi.errors import ApiError, ConfigError, UnreachableError, WaitTimeout
from stratafi.resources import (
    Corpora,
    Evals,
    EvalSets,
    Experiments,
    Maps,
    Models,
    Pipelines,
    Profiles,
    RetrievalConfigs,
    Runs,
    Tests,
    Training,
)
from stratafi.results import Reports
from stratafi.version import __version__

API_PREFIX = "/api/v1"
IDENTITY_HEADER = "X-Stratafi-Identity"
API_TOKEN_PREFIX = "sft_"  # noqa: S105 - the prefix that marks an API token, not one
# Who ``from_env`` signs in as when the environment names no token and no sign-in.
DEFAULT_DEV_SIGN_IN = "local:owner"
FINISHED = frozenset({"succeeded", "stopped", "failed", "cancelled"})

GET_ATTEMPTS = 3
RETRY_STATUSES = frozenset({429, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 30.0
RENEW_BEFORE_SECONDS = 60.0
BACKOFF_START_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 15.0
RECHECK_SECONDS = 30.0
POLL_SECONDS = 2.0
# The API sends a keepalive comment every 15 seconds, so a read that waits longer is a dead link.
STREAM_TIMEOUT = httpx2.Timeout(10.0, read=45.0)

_DEV_SIGN_IN_RE = re.compile(r"^(?P<tenant>[a-z0-9_]{1,64}):(?P<role>owner|editor|viewer|guest)$")


@dataclass
class Event:
    """One server-sent event: ``kind`` (``run``, ``eval_run``, ``job``, ``lagged``) and its
    JSON data."""

    kind: str
    data: dict[str, Any]


def sse_events(lines: Iterable[str]) -> Iterator[Event | None]:
    """Server-sent events from a stream's lines. Comments (the API's keepalives) yield
    ``None``, so a reader wakes up at least that often."""
    kind, data = "message", []
    for line in lines:
        if not line:
            if data:
                try:
                    payload = json.loads("\n".join(data))
                except ValueError:
                    payload = {}
                yield Event(kind, payload if isinstance(payload, dict) else {})
            kind, data = "message", []
        elif line.startswith(":"):
            yield None
        else:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                kind = value
            elif field == "data":
                data.append(value)


@dataclass
class _DevSignIn:
    tenant: str
    role: str
    token: str | None = None
    renew_at: float = 0.0


def _dev_sign_in(value: str) -> _DevSignIn:
    match = _DEV_SIGN_IN_RE.fullmatch(value.strip())
    if match is None:
        raise ConfigError("STRATAFI_DEV_SIGN_IN must be tenant:role, such as tenant0:editor")
    return _DevSignIn(match["tenant"], match["role"])


def _retry_after(response: httpx2.Response, default: float) -> float:
    try:
        return min(float(response.headers.get("Retry-After", "")), MAX_RETRY_AFTER_SECONDS)
    except ValueError:
        return default


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def _query(params: Mapping[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (params or {}).items() if value is not None}


class Client:
    """The Stratafi API. ``Client.from_env()`` reads the environment; ``Client(url, token=...)``
    takes it directly. ``transport`` replaces the HTTP transport (tests)."""

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        dev_sign_in: str | None = None,
        timeout: float = 60.0,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        base = (url or "").strip().rstrip("/")
        if not re.match(r"^https?://[^/\s]+", base):
            raise ConfigError("the API's address must be an http:// or https:// URL")
        self.url = base if base.endswith(API_PREFIX) else base + API_PREFIX
        self._token = token or None
        self._dev = _dev_sign_in(dev_sign_in) if dev_sign_in and not token else None
        self._http = httpx2.Client(
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": f"stratafi-sdk/{__version__}"},
        )
        self.corpora = Corpora(self)
        self.pipelines = Pipelines(self)
        self.retrieval = RetrievalConfigs(self)
        self.runs = Runs(self)
        self.eval_sets = EvalSets(self)
        self.evals = Evals(self)
        self.experiments = Experiments(self)
        self.profiles = Profiles(self)
        self.tests = Tests(self)
        self.training = Training(self)
        self.models = Models(self)
        self.maps = Maps(self)
        self.reports = Reports()

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        url: str | None = None,
        **kwargs: Any,
    ) -> "Client":
        """A client configured by ``STRATAFI_URL`` and ``STRATAFI_TOKEN`` (or
        ``STRATAFI_DEV_SIGN_IN``, ``local:owner`` when neither is set); ``url`` overrides the
        first."""
        env = os.environ if environ is None else environ
        address = url or env.get("STRATAFI_URL", "")
        if not address.strip():
            raise ConfigError(
                "set STRATAFI_URL to the API's address, such as http://localhost:8100"
            )
        token = env.get("STRATAFI_TOKEN") or None
        dev_sign_in = env.get("STRATAFI_DEV_SIGN_IN") or (None if token else DEFAULT_DEV_SIGN_IN)
        return cls(address, token=token, dev_sign_in=dev_sign_in, **kwargs)

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Client({self.url!r})"

    def close(self) -> None:
        self._http.close()

    # --- HTTP ------------------------------------------------------------------------------

    def _identity(self) -> dict[str, str]:
        if self._token and self._token.startswith(API_TOKEN_PREFIX):
            return {"Authorization": f"Bearer {self._token}"}
        if self._token:
            return {IDENTITY_HEADER: self._token}
        if self._dev is not None:
            return {IDENTITY_HEADER: self._dev_token()}
        return {}

    def _dev_token(self) -> str:
        dev = self._dev
        if dev is None:
            raise ConfigError("no development sign-in is configured")
        if dev.token is None or time.monotonic() >= dev.renew_at:
            try:
                response = self._http.post(
                    f"{self.url}/dev/token", json={"tenant": dev.tenant, "role": dev.role}
                )
            except httpx2.TransportError as error:
                raise self._unreachable(error) from None
            if response.status_code == 404:
                raise ConfigError(
                    "the API has no development sign-in (no identity.dev_sign_in_key, or it runs "
                    "in prod): set STRATAFI_TOKEN instead"
                )
            if response.status_code >= 400:
                raise self._problem(response)
            body = response.json()
            lifetime = float(body["expires_in"])
            dev.token = str(body["token"])
            dev.renew_at = time.monotonic() + max(lifetime - RENEW_BEFORE_SECONDS, lifetime / 2)
        return dev.token

    def _unreachable(self, error: Exception) -> UnreachableError:
        return UnreachableError(f"the API at {self.url} didn't answer ({type(error).__name__})")

    @staticmethod
    def _problem(response: httpx2.Response) -> ApiError:
        try:
            problem = response.json()
        except ValueError:
            problem = {}
        if not isinstance(problem, dict):
            problem = {}
        problem.setdefault("title", response.reason_phrase or "Error")
        return ApiError(response.status_code, problem)

    def send(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        files: Any = None,
    ) -> tuple[int, Any]:
        """One API call: its status code and JSON body. GETs are retried on transient
        failures; an error status raises ``ApiError``."""
        response = self._request(method, path, params=params, json=json, files=files)
        return response.status_code, (response.json() if response.content else None)

    def get_text(self, path: str, **params: Any) -> str:
        """A GET whose answer is text, such as a profile's YAML."""
        return self._request("GET", path, params=params).text

    def get_bytes(self, path: str, **params: Any) -> bytes:
        """A GET whose answer is a file, such as a trained model's zip."""
        return self._request("GET", path, params=params).content

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        files: Any = None,
    ) -> httpx2.Response:
        attempts = GET_ATTEMPTS if method == "GET" else 1
        delay = BACKOFF_START_SECONDS
        for attempt in range(1, attempts + 1):
            try:
                response = self._http.request(
                    method,
                    f"{self.url}{path}",
                    params=_query(params),
                    json=json,
                    files=files,
                    headers=self._identity(),
                )
            except httpx2.TransportError as error:
                if attempt == attempts:
                    raise self._unreachable(error) from None
            else:
                if response.status_code not in RETRY_STATUSES or attempt == attempts:
                    break
                delay = _retry_after(response, delay)
            _pause(delay)
            delay = min(delay * 2, BACKOFF_MAX_SECONDS)
        if response.status_code >= 400:
            raise self._problem(response)
        return response

    def get(self, path: str, **params: Any) -> Any:
        return self.send("GET", path, params=params)[1]

    def post(self, path: str, body: Any = None, **params: Any) -> Any:
        return self.send("POST", path, params=params, json=body)[1]

    def put(self, path: str, body: Any = None, **params: Any) -> Any:
        return self.send("PUT", path, params=params, json=body)[1]

    def patch(self, path: str, body: Any = None, **params: Any) -> Any:
        return self.send("PATCH", path, params=params, json=body)[1]

    def delete(self, path: str, **params: Any) -> Any:
        return self.send("DELETE", path, params=params)[1]

    def pages(self, path: str, **params: Any) -> Iterator[dict[str, Any]]:
        """Every item of a paginated list (``items`` and ``next_cursor``)."""
        cursor = None
        while True:
            page = self.get(path, limit=200, cursor=cursor, **params)
            yield from page["items"]
            cursor = page.get("next_cursor")
            if not cursor:
                return

    # --- waiting ---------------------------------------------------------------------------

    def follow(
        self,
        kind: str,
        run_id: str,
        read: Callable[[], Any],
        *,
        timeout: float | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        finished: frozenset[str] = FINISHED,
    ) -> Any:
        """Follow a run's events (``kind`` ``run``, ``eval_run``, ``experiment`` or
        ``test_run``) until it reaches a status in ``finished``; the final record, as ``read()``
        returns it. Raises ``WaitTimeout`` after ``timeout`` seconds."""
        deadline = None if timeout is None else time.monotonic() + timeout
        delay = BACKOFF_START_SECONDS
        polling = False

        def left() -> float | None:
            if deadline is None:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WaitTimeout(f"{kind.replace('_', ' ')} {run_id} hasn't finished yet")
            return remaining

        while True:
            record = read()
            if record.status in finished:
                return record
            remaining = left()
            if polling:
                _pause(min(POLL_SECONDS, remaining or POLL_SECONDS))
                continue
            try:
                with self._http.stream(
                    "GET",
                    f"{self.url}/events",
                    params={"run_id": run_id},
                    headers=self._identity(),
                    timeout=STREAM_TIMEOUT,
                ) as response:
                    if response.status_code in (404, 501, 503):
                        polling = True  # no events here: poll instead
                        continue
                    if response.status_code >= 400:
                        response.read()
                        raise self._problem(response)
                    delay = BACKOFF_START_SECONDS
                    record = read()  # subscribed now, so nothing slips between
                    if record.status in finished:
                        return record
                    checked = time.monotonic()
                    for event in sse_events(response.iter_lines()):
                        left()
                        recheck = time.monotonic() - checked >= RECHECK_SECONDS
                        if event is not None and event.kind == "lagged":
                            recheck = True
                        elif event is not None and event.kind == kind:
                            if event.data.get("run_id") != run_id:
                                continue
                            if on_update is not None:
                                on_update(event.data)
                            recheck = recheck or event.data.get("status") in finished
                        if recheck:
                            record = read()
                            checked = time.monotonic()
                            if record.status in finished:
                                return record
            except httpx2.TransportError:
                pass  # the stream broke: reconnect after a pause
            remaining = left()
            _pause(min(delay, remaining) if remaining is not None else delay)
            delay = min(delay * 2, BACKOFF_MAX_SECONDS)
