"""Targets for sandbox tests (``stratafi_components.sandbox``): each misbehaves in one way."""

import os
import time
from typing import Any


def echo(payload: dict[str, Any]) -> dict[str, Any]:
    import tempfile

    return {
        "payload": payload,
        "cwd": os.getcwd(),
        "env": sorted(os.environ),
        "tmpdir": tempfile.gettempdir(),
    }


def hang(payload: dict[str, Any]) -> None:
    time.sleep(float(payload.get("seconds", 3600)))


def allocate(payload: dict[str, Any]) -> int:
    block = bytearray(int(payload["bytes"]))
    return len(block)


def crash(payload: dict[str, Any]) -> None:
    os._exit(9)


def refuse(payload: dict[str, Any]) -> None:
    from stratafi_core.components import ComponentError

    raise ComponentError("encrypted")


def leak(payload: dict[str, Any]) -> None:
    raise ValueError(f"the document said: {payload['secret']}")


def reach(payload: dict[str, Any]) -> dict[str, bool]:
    """Every way out a hostile parser might take: the confinement's own probe."""
    from stratafi_components.confine import probe

    return probe(payload)
