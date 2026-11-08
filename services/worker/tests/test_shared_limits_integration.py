"""Provider limits shared by every worker process: two limiters, as two
workers, take from one quota and count each other's calls in flight."""

import threading
import time

import pytest
import sqlalchemy as sa

from stratafi_core.settings import ProviderLimit
from stratafi_store.scope import system_scope
from stratafi_worker.limits import SharedProviderLimiter, WaitCancelled

pytestmark = pytest.mark.integration


def test_two_workers_share_one_request_and_token_quota(migrated_engine: sa.Engine) -> None:
    limits = {"gateway": ProviderLimit(requests_per_minute=6, tokens_per_minute=600)}
    first = SharedProviderLimiter(migrated_engine, limits, holder="worker-a")
    second = SharedProviderLimiter(migrated_engine, limits, holder="worker-b")
    for index in range(6):  # a minute's six requests, whichever worker asks
        lease, wait = (first if index % 2 else second)._try("gateway", limits["gateway"], 50)
        assert lease is not None
        assert wait == 0.0
        (first if index % 2 else second)._release(lease)
    lease, wait = first._try("gateway", limits["gateway"], 50)
    assert lease is None
    assert 5 < wait <= 10.5  # the seventh waits for the bucket: 0.1 a second
    # Tokens too: 300 were taken; 600 more won't fit.
    with system_scope(), migrated_engine.connect() as conn:
        level = conn.execute(sa.text("SELECT tokens FROM stratafi.provider_quota")).scalar_one()
    assert 299 < level < 301


def test_concurrency_counts_every_workers_calls(migrated_engine: sa.Engine) -> None:
    limits = {"gateway": ProviderLimit(concurrency=2)}
    limiters = [
        SharedProviderLimiter(migrated_engine, limits, holder=f"worker-{n}", poll_seconds=0.02)
        for n in range(3)
    ]
    running, peak, lock = 0, 0, threading.Lock()

    def call(limiter: SharedProviderLimiter) -> None:
        nonlocal running, peak
        with limiter.acquire("gateway"):
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.2)
            with lock:
                running -= 1

    threads = [threading.Thread(target=call, args=(limiters[n % 3],)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == 2
    with system_scope(), migrated_engine.connect() as conn:
        assert (
            conn.execute(sa.text("SELECT count(*) FROM stratafi.provider_leases")).scalar_one() == 0
        )


def test_a_cancelled_job_stops_waiting_and_unlimited_providers_pass(
    migrated_engine: sa.Engine,
) -> None:
    limits = {"gateway": ProviderLimit(concurrency=1)}
    holder = SharedProviderLimiter(migrated_engine, limits, holder="a")
    waiter = SharedProviderLimiter(migrated_engine, limits, holder="b", poll_seconds=0.02)
    cancelled = threading.Event()
    with holder.acquire("gateway"):
        threading.Timer(0.1, cancelled.set).start()
        with pytest.raises(WaitCancelled), waiter.acquire("gateway", cancelled=cancelled):
            pass
    with waiter.acquire("other-provider") as waited:
        assert waited == 0.0
