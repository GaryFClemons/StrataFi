"""Limits per remote provider: requests per minute, tokens per minute, and concurrency, so a
sweep can't exhaust a provider's quota.

``ProviderLimiter`` holds them in one process; ``SharedProviderLimiter`` shares them
between every worker process through Postgres, which ``providers.shared_limits`` turns on (the
default), so adding workers never multiplies a quota. Waits count as rate-limit time for the
run record, and a cancelled job stops waiting.
"""

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from stratafi_core.settings import ProviderLimit


class WaitCancelled(Exception):
    """The job was cancelled while waiting for a provider."""


class TokenBucket:
    """``capacity`` units, refilled continuously at ``per_second``."""

    def __init__(
        self, capacity: float, per_second: float, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.capacity = capacity
        self.per_second = per_second
        self._clock = clock
        self._level = capacity
        self._at = clock()
        self._lock = threading.Lock()

    def take(self, amount: float) -> float:
        """Take ``amount`` if available and return 0; otherwise take nothing and return how long
        to wait before it would be."""
        amount = min(amount, self.capacity)  # a request bigger than the bucket waits for a full one
        with self._lock:
            now = self._clock()
            self._level = min(self.capacity, self._level + (now - self._at) * self.per_second)
            self._at = now
            if self._level >= amount:
                self._level -= amount
                return 0.0
            return (amount - self._level) / self.per_second


class _Provider:
    def __init__(self, limit: ProviderLimit, clock: Callable[[], float]) -> None:
        self.requests = (
            TokenBucket(limit.requests_per_minute, limit.requests_per_minute / 60, clock=clock)
            if limit.requests_per_minute
            else None
        )
        self.tokens = (
            TokenBucket(limit.tokens_per_minute, limit.tokens_per_minute / 60, clock=clock)
            if limit.tokens_per_minute
            else None
        )
        self.slots = threading.BoundedSemaphore(limit.concurrency) if limit.concurrency else None


class ProviderLimiter:
    def __init__(
        self,
        limits: Mapping[str, ProviderLimit],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._providers = {name: _Provider(limit, clock) for name, limit in limits.items()}
        self._sleep = sleep

    @contextmanager
    def acquire(
        self, provider: str, *, tokens: int = 0, cancelled: threading.Event | None = None
    ) -> Iterator[float]:
        """Hold a slot for one call to ``provider``. Yields the seconds spent waiting."""
        state = self._providers.get(provider)
        if state is None:
            yield 0.0
            return
        cancelled = cancelled or threading.Event()
        waited = 0.0
        if state.slots is not None:
            while not state.slots.acquire(timeout=0.1):
                waited += 0.1
                if cancelled.is_set():
                    raise WaitCancelled(provider)
        try:
            for bucket, amount in ((state.requests, 1), (state.tokens, tokens)):
                if bucket is None or amount <= 0:
                    continue
                while (delay := bucket.take(amount)) > 0:
                    if cancelled.is_set():
                        raise WaitCancelled(provider)
                    pause = min(delay, 0.25)
                    if self._sleep is not None:
                        self._sleep(pause)
                    elif cancelled.wait(pause):
                        raise WaitCancelled(provider)
                    waited += pause
            yield waited
        finally:
            if state.slots is not None:
                state.slots.release()


class SharedProviderLimiter(ProviderLimiter):
    """``providers.limits`` shared by every worker process through Postgres.

    Each call takes from the provider's buckets in ``provider_quota`` under its row's lock,
    refilling them by the time since the last take, and holds a lease in ``provider_leases``
    while it runs; concurrency counts the unexpired leases of every process. A lease lapses
    after ``lease_seconds``, so a dead worker's calls stop counting. Waits poll every
    ``poll_seconds``. Same interface as the per-process limiter, which it replaces."""

    def __init__(
        self,
        engine: Any,
        limits: Mapping[str, ProviderLimit],
        *,
        holder: str = "worker",
        lease_seconds: float = 300.0,
        poll_seconds: float = 0.05,
    ) -> None:
        super().__init__({})
        self.engine = engine
        self.limits = dict(limits)
        self.holder = holder[:120]
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds

    def _try(self, provider: str, limit: ProviderLimit, tokens: int) -> tuple[Any, float]:
        """A lease, or None and how long to wait before trying again."""
        import uuid

        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert

        from stratafi_store.scope import system_scope
        from stratafi_store.tables import provider_leases, provider_quota

        rpm, tpm = limit.requests_per_minute, limit.tokens_per_minute
        with system_scope(), self.engine.begin() as conn:
            conn.execute(
                insert(provider_quota)
                .values(provider=provider, requests=rpm, tokens=tpm)
                .on_conflict_do_nothing()
            )
            row = conn.execute(
                sa.select(
                    provider_quota.c.requests,
                    provider_quota.c.tokens,
                    sa.func.extract("epoch", sa.func.now() - provider_quota.c.refilled_at),
                )
                .where(provider_quota.c.provider == provider)
                .with_for_update()
            ).one()
            elapsed = max(float(row[2]), 0.0)
            requests = (
                min(rpm, (row[0] if row[0] is not None else rpm) + elapsed * rpm / 60)
                if rpm
                else None
            )
            amount = min(tokens, tpm) if tpm else 0
            level = (
                min(tpm, (row[1] if row[1] is not None else tpm) + elapsed * tpm / 60)
                if tpm
                else None
            )
            if limit.concurrency:
                busy = conn.execute(
                    sa.select(sa.func.count())
                    .select_from(provider_leases)
                    .where(
                        provider_leases.c.provider == provider,
                        provider_leases.c.expires_at > sa.func.now(),
                    )
                ).scalar_one()
                if busy >= limit.concurrency:
                    return None, self.poll_seconds
            wait = 0.0
            if requests is not None and requests < 1:
                wait = max(wait, (1 - requests) / (rpm / 60))
            if level is not None and amount > 0 and level < amount:
                wait = max(wait, (amount - level) / (tpm / 60))
            if wait > 0:
                return None, wait
            conn.execute(
                sa.update(provider_quota)
                .where(provider_quota.c.provider == provider)
                .values(
                    requests=None if requests is None else requests - 1,
                    tokens=None if level is None else level - amount,
                    refilled_at=sa.func.now(),
                )
            )
            lease = uuid.uuid4()
            conn.execute(
                sa.insert(provider_leases).values(
                    id=lease,
                    provider=provider,
                    holder=self.holder,
                    expires_at=sa.func.now()
                    + sa.text(f"interval '{int(self.lease_seconds)} second'"),
                )
            )
            return lease, 0.0

    def _release(self, lease: Any) -> None:
        import sqlalchemy as sa

        from stratafi_store.scope import system_scope
        from stratafi_store.tables import provider_leases

        with system_scope(), self.engine.begin() as conn:
            conn.execute(sa.delete(provider_leases).where(provider_leases.c.id == lease))

    @contextmanager
    def acquire(
        self, provider: str, *, tokens: int = 0, cancelled: threading.Event | None = None
    ) -> Iterator[float]:
        limit = self.limits.get(provider)
        if limit is None:
            yield 0.0
            return
        cancelled = cancelled or threading.Event()
        waited = 0.0
        while True:
            lease, wait = self._try(provider, limit, tokens)
            if lease is not None:
                break
            pause = min(max(wait, self.poll_seconds), 0.25)
            if cancelled.wait(pause):
                raise WaitCancelled(provider)
            waited += pause
        try:
            yield waited
        finally:
            self._release(lease)
