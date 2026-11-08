"""Live events from Postgres ``LISTEN``/``NOTIFY``.

Writers notify ``stratafi_events`` inside their transaction (``stratafi_store.jobs``), so a
notification arrives only once its change is committed. ``EventListener`` holds one listening
connection in a background thread and hands each event (a JSON object) to ``deliver``. It uses
a synchronous connection, like the rest of the store, so it works on Windows too. If the
connection drops it reconnects, backing off; events sent while it was down are lost, so
consumers treat events as hints and read state from the API.
"""

import json
import logging
import threading
from collections.abc import Callable
from typing import Any

import psycopg
from psycopg import sql

from stratafi_store.jobs import EVENTS_CHANNEL

logger = logging.getLogger("stratafi.store.events")


class EventListener:
    def __init__(
        self,
        url: str,
        deliver: Callable[[dict[str, Any]], None],
        *,
        channel: str = EVENTS_CHANNEL,
        poll_seconds: float = 0.5,
        max_backoff_seconds: float = 10.0,
    ) -> None:
        self.url = url
        self.deliver = deliver
        self.channel = channel
        self.poll_seconds = poll_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.listening = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="stratafi-events", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                with psycopg.connect(self.url, autocommit=True) as conn:
                    conn.execute(sql.SQL("LISTEN {}").format(sql.Identifier(self.channel)))
                    self.listening.set()
                    backoff = 0.5
                    while not self._stop.is_set():
                        for notify in conn.notifies(timeout=self.poll_seconds, stop_after=100):
                            self._deliver(notify.payload)
            except psycopg.Error as exc:
                self.listening.clear()
                logger.warning(
                    "event listener disconnected", extra={"exc_kind": type(exc).__name__}
                )
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.max_backoff_seconds)
        self.listening.clear()

    def _deliver(self, payload: str) -> None:
        try:
            event = json.loads(payload)
        except ValueError:
            logger.warning("event payload isn't JSON")
            return
        if isinstance(event, dict):
            try:
                self.deliver(event)
            except Exception:
                logger.exception("event delivery failed")
