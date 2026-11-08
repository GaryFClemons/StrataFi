"""Fan live events out to server-sent-event streams.

One ``EventListener`` per API process listens to ``stratafi_events`` (``LISTEN``/``NOTIFY``) and
starts with the first subscriber. Each subscriber gets the events of its own workspace in an
``asyncio.Queue``; a subscriber that falls more than ``MAX_PENDING`` events behind loses the
oldest and is told it lagged, since events are hints and state lives in the API.
"""

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from stratafi_store.events import EventListener

MAX_PENDING = 1000


@dataclass(eq=False)
class Subscription:
    workspace_id: uuid.UUID
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=lambda: asyncio.Queue(MAX_PENDING))
    lagged: bool = False

    def offer(self, event: dict[str, Any]) -> None:
        """Runs on the event loop's thread."""
        if self.queue.full():
            self.queue.get_nowait()
            self.lagged = True
        self.queue.put_nowait(event)


class EventHub:
    def __init__(self, url: str) -> None:
        self._listener = EventListener(url, self._deliver)
        self._subscriptions: set[Subscription] = set()
        self._lock = threading.Lock()

    @property
    def listening(self) -> threading.Event:
        return self._listener.listening

    def subscribe(self, workspace_id: uuid.UUID) -> Subscription:
        subscription = Subscription(workspace_id, asyncio.get_running_loop())
        with self._lock:
            self._subscriptions.add(subscription)
        self._listener.start()
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            self._subscriptions.discard(subscription)

    def stop(self) -> None:
        self._listener.stop()

    def _deliver(self, event: dict[str, Any]) -> None:
        """Runs on the listener's thread."""
        workspace = event.get("workspace_id")
        with self._lock:
            targets = [s for s in self._subscriptions if str(s.workspace_id) == workspace]
        for subscription in targets:
            try:
                subscription.loop.call_soon_threadsafe(subscription.offer, event)
            except RuntimeError:
                self.unsubscribe(subscription)  # its event loop has closed
