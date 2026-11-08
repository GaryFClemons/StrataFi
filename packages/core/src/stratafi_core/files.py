"""Small file helpers."""

import os
from pathlib import Path


def write_private_file(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path``, readable only by the owner on Linux (mode 0600).

    Windows ignores the mode; there the file inherits the folder's permissions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
