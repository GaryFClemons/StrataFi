"""The worker's automation: it fires suites' schedules with its heartbeats
(``stratafi_store.automation.SuiteScheduler``), and sends webhook deliveries
(``webhook.deliver``).

A delivery goes to the webhook's URL through the same SSRF guard as URL import
(``stratafi_components.fetch.UrlGuard``): the host's every address must be public, unless
``suites.webhook_allow_private`` says otherwise for development, and the connection goes to the
checked address, with no redirects and no proxy. It's signed when it's sent
(``stratafi_core.webhooks``). A 2xx answer delivers it; a timeout, an unreachable host, a 408,
a 429 or a 5xx is retried by the job queue with backoff, up to ``suites.webhook_attempts``; any
other answer, or a refused URL, fails it at once. The URL and the body never reach a log.
"""

import uuid
from collections.abc import Mapping
from typing import Any

import httpx2
from sqlalchemy.engine import Engine

from stratafi_components.fetch import FetchRefused, UrlGuard
from stratafi_core.settings import Settings
from stratafi_core.webhooks import (
    DELIVERY_HEADER,
    EVENT_HEADER,
    SIGNATURE_HEADER,
    signature_header,
)
from stratafi_store import webhooks
from stratafi_store.automation import SuiteScheduler
from stratafi_store.jobs import JobQueue
from stratafi_worker.worker import HandlerRegistry, JobContext, PermanentJobError

RETRY_STATUSES = frozenset({408, 429})
USER_AGENT = "Stratafi-Webhooks/1"


class DeliveryFailed(Exception):
    """A send that may succeed later: the job queue retries it."""


class Deliverer:
    """The ``webhook.deliver`` job."""

    def __init__(
        self, engine: Engine, settings: Settings, transport: httpx2.BaseTransport | None = None
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.transport = transport
        suites = settings.suites
        self.guard = UrlGuard(
            allowed_ports=suites.webhook_ports, allow_private=suites.webhook_allow_private
        )

    def __call__(self, job: JobContext) -> Mapping[str, Any]:
        delivery_id = uuid.UUID(str(job.payload["delivery_id"]))
        key = self.settings.suites.webhook_secret_key
        begun = webhooks.begin_delivery(
            self.engine, delivery_id, secret_key=key.get_secret_value() if key else None
        )
        if begun is None:
            return {"skipped": True}
        delivery, url, secret = begun
        final = job.job.attempts >= job.job.max_attempts
        try:
            status = self._send(url, secret, delivery)
        except FetchRefused as refused:
            webhooks.finish_delivery(
                self.engine, delivery_id, delivered=False, error_code=refused.code
            )
            raise PermanentJobError(refused.code) from None
        except httpx2.TimeoutException:
            return self._retry(delivery_id, "timeout", None, final)
        except httpx2.HTTPError:
            return self._retry(delivery_id, "unreachable", None, final)
        if 200 <= status < 300:
            webhooks.finish_delivery(
                self.engine, delivery_id, delivered=True, response_status=status
            )
            return {"delivered": True, "status": status}
        code = f"http_{status}"
        if status >= 500 or status in RETRY_STATUSES:
            return self._retry(delivery_id, code, status, final)
        webhooks.finish_delivery(
            self.engine, delivery_id, delivered=False, response_status=status, error_code=code
        )
        raise PermanentJobError(code)

    def _retry(
        self, delivery_id: uuid.UUID, code: str, status: int | None, final: bool
    ) -> Mapping[str, Any]:
        webhooks.finish_delivery(
            self.engine,
            delivery_id,
            delivered=False,
            response_status=status,
            error_code=code,
            final=final,
        )
        if final:
            return {"delivered": False, "error": code}
        raise DeliveryFailed(code)

    def _send(self, url: str, secret: str, delivery: webhooks.Delivery) -> int:
        target = self.guard.check(url)
        body = delivery.body.encode("utf-8")
        with httpx2.Client(
            timeout=self.settings.suites.webhook_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=self.transport,
        ) as client:
            request = client.build_request(
                "POST",
                target.connect_url(),
                content=body,
                headers={
                    "Host": target.host_header(),
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT,
                    EVENT_HEADER: delivery.event,
                    DELIVERY_HEADER: str(delivery.id),
                    SIGNATURE_HEADER: signature_header(secret, body),
                },
                extensions={"sni_hostname": target.host} if target.scheme == "https" else {},
            )
            response = client.send(request)
            response.close()
        return response.status_code


def register_automation_handlers(
    handlers: HandlerRegistry, *, engine: Engine, queue: JobQueue, settings: Settings
) -> SuiteScheduler:
    """Register ``webhook.deliver``; returns the scheduler, whose ``tick`` the worker runs with
    its heartbeats."""
    handlers.register(webhooks.DELIVER_JOB, Deliverer(engine, settings))
    return SuiteScheduler(engine, queue, max_attempts=settings.worker.max_attempts)
