"""Run an ASGI app with Uvicorn on 127.0.0.1 and a free port, in a thread: for tests that need a
real HTTP server, such as server-sent events."""

import threading
import time
from typing import Any

import uvicorn


class ServerThread:
    def __init__(self, app: Any) -> None:
        config = uvicorn.Config(
            app, host="127.0.0.1", port=0, log_config=None, access_log=False, lifespan="on"
        )
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self.server.run, name="test-server", daemon=True)
        self.url = ""

    def start(self, timeout: float = 10.0) -> "ServerThread":
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self.server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("the test server didn't start")
            time.sleep(0.01)
        host, port = self.server.servers[0].sockets[0].getsockname()[:2]
        self.url = f"http://{host}:{port}"
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self.server.should_exit = True
        self._thread.join(timeout)
