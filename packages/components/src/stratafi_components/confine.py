"""Confine a sandboxed parse: no network, no reaching the worker, a read-only filesystem.

``confine(...)`` runs in the sandbox's child process before the parser is imported. On Linux it
applies, in order, as much of this as the kernel allows:

1. **A network namespace** (``unshare(CLONE_NEWNET)``, or with a new user namespace when the
   process lacks ``CAP_SYS_ADMIN``): the child has only its own, unconfigured loopback.
2. **Landlock** (Linux 5.13 and later): reading is limited to Python's installation, the
   import path, the system's libraries and data (``/usr``, ``/lib``, ``/etc``...) and the one
   document; writing to the scratch directory. Secrets elsewhere (``.env``, ``data/keys``,
   ``/run/secrets``, the sessions file) can't be opened, nor another process's
   ``/proc``. From ABI 4 it refuses TCP too, and
   from ABI 6 signals and abstract sockets outside the sandbox.
3. **seccomp**, with ``no_new_privs``: ``socket``, ``ptrace``, ``process_vm_readv`` and
   ``_writev``, io_uring, ``bpf``, the keyrings, ``userfaultfd``, ``perf_event_open``, mounts
   and new namespaces fail with ``EPERM`` (``clone3`` with ``ENOSYS``, so the C library falls
   back to ``clone``, whose namespace flags are checked). Another architecture's system calls,
   such as x32's, kill the process.

Without seccomp nothing stops a socket, so ``limits.parse_confinement: required`` (which a prod
deployment must set) fails the parse with ``sandbox_unconfined`` instead. The worker also makes
itself undumpable (``harden_process``), so a parser running as the same user can't read its
memory or environment through ``/proc``.

``Confinement`` says what applied; a run records it with the parse's metrics.
"""

import contextlib
import ctypes
import os
import platform
import struct
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# --- what applied ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Confinement:
    network: str = "none"  # "namespace", "seccomp" or "none"
    seccomp: bool = False
    landlock: int = 0  # the Landlock ABI applied, 0 if none

    @property
    def blocks_network(self) -> bool:
        return self.seccomp or self.network == "namespace"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# --- system calls ------------------------------------------------------------------------------

_ARCHES: dict[str, dict[str, int]] = {
    # AUDIT_ARCH_*, then the numbers from each architecture's unistd table.
    "x86_64": {
        "arch": 0xC000003E,
        "socket": 41,
        "clone": 56,
        "ptrace": 101,
        "pivot_root": 155,
        "mount": 165,
        "umount2": 166,
        "init_module": 175,
        "delete_module": 176,
        "kexec_load": 246,
        "add_key": 248,
        "request_key": 249,
        "keyctl": 250,
        "unshare": 272,
        "perf_event_open": 298,
        "setns": 308,
        "process_vm_readv": 310,
        "process_vm_writev": 311,
        "finit_module": 313,
        "bpf": 321,
        "userfaultfd": 323,
        "io_uring_setup": 425,
        "io_uring_enter": 426,
        "io_uring_register": 427,
        "clone3": 435,
    },
    "aarch64": {
        "arch": 0xC00000B7,
        "socket": 198,
        "clone": 220,
        "ptrace": 117,
        "pivot_root": 41,
        "mount": 40,
        "umount2": 39,
        "init_module": 105,
        "delete_module": 106,
        "kexec_load": 104,
        "add_key": 217,
        "request_key": 218,
        "keyctl": 219,
        "unshare": 97,
        "perf_event_open": 241,
        "setns": 268,
        "process_vm_readv": 270,
        "process_vm_writev": 271,
        "finit_module": 273,
        "bpf": 280,
        "userfaultfd": 282,
        "io_uring_setup": 425,
        "io_uring_enter": 426,
        "io_uring_register": 427,
        "clone3": 435,
    },
}
_LANDLOCK_CREATE_RULESET, _LANDLOCK_ADD_RULE, _LANDLOCK_RESTRICT_SELF = 444, 445, 446

DENIED = (
    "socket",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "bpf",
    "add_key",
    "request_key",
    "keyctl",
    "userfaultfd",
    "perf_event_open",
    "mount",
    "umount2",
    "pivot_root",
    "unshare",
    "setns",
    "kexec_load",
    "init_module",
    "finit_module",
    "delete_module",
)
# CLONE_NEWNS, NEWCGROUP, NEWUTS, NEWIPC, NEWUSER, NEWPID, NEWNET
_CLONE_NAMESPACES = 0x7E020000
_X32_SYSCALL_BIT = 0x40000000

_PR_SET_DUMPABLE, _PR_SET_SECCOMP, _PR_SET_NO_NEW_PRIVS = 4, 22, 38
_SECCOMP_MODE_FILTER = 2
_RET_KILL_PROCESS, _RET_ERRNO, _RET_ALLOW = 0x80000000, 0x00050000, 0x7FFF0000
_EPERM, _ENOSYS = 1, 38

# Classic BPF opcodes.
_LD_W_ABS, _JEQ_K, _JGE_K, _JSET_K, _RET_K = 0x20, 0x15, 0x35, 0x45, 0x06


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _arch() -> dict[str, int] | None:
    if not sys.platform.startswith("linux"):
        return None
    machine = platform.machine().lower()
    return _ARCHES.get({"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine))


def seccomp_program(arch: dict[str, int], *, x32: bool) -> bytes:
    """The filter as classic BPF: a list of ``struct sock_filter``."""
    code: list[tuple[int, int, object, object]] = []  # (op, k, true target, false target)

    def emit(op: int, k: int, true: object = 0, false: object = 0) -> None:
        code.append((op, k, true, false))

    emit(_LD_W_ABS, 4)  # seccomp_data.arch
    emit(_JEQ_K, arch["arch"], 1, 0)
    emit(_RET_K, _RET_KILL_PROCESS)
    emit(_LD_W_ABS, 0)  # seccomp_data.nr
    if x32:
        emit(_JGE_K, _X32_SYSCALL_BIT, "eperm", 0)
    for name in DENIED:
        emit(_JEQ_K, arch[name], "eperm", 0)
    emit(_JEQ_K, arch["clone3"], "enosys", 0)
    emit(_JEQ_K, arch["clone"], 0, "allow")
    emit(_LD_W_ABS, 16)  # the low half of seccomp_data.args[0]: clone's flags
    emit(_JSET_K, _CLONE_NAMESPACES, "eperm", "allow")
    labels = {"allow": len(code), "eperm": len(code) + 1, "enosys": len(code) + 2}
    emit(_RET_K, _RET_ALLOW)
    emit(_RET_K, _RET_ERRNO | _EPERM)
    emit(_RET_K, _RET_ERRNO | _ENOSYS)

    def offset(index: int, target: object) -> int:
        if isinstance(target, str):
            return labels[target] - index - 1
        return int(target)  # type: ignore[call-overload]

    return b"".join(
        struct.pack("=HBBI", op, offset(i, true), offset(i, false), k)
        for i, (op, k, true, false) in enumerate(code)
    )


def _no_new_privs(libc: ctypes.CDLL) -> bool:
    return libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0


def install_seccomp() -> bool:
    """Install the filter on this process (and what it starts). False if it can't be."""
    arch = _arch()
    if arch is None:
        return False
    program = seccomp_program(arch, x32=arch is _ARCHES["x86_64"])
    buffer = ctypes.create_string_buffer(program, len(program))
    fprog = _SockFprog(len(program) // 8, ctypes.addressof(buffer))
    libc = _libc()
    if not _no_new_privs(libc):
        return False
    return libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) == 0


# --- network namespace -------------------------------------------------------------------------


def isolate_network() -> bool:
    """Move this process into a new network namespace: as root, or through a new user
    namespace. False if neither is allowed."""
    unshare = getattr(os, "unshare", None)
    if unshare is None:
        return False
    for flags in (os.CLONE_NEWNET, os.CLONE_NEWUSER | os.CLONE_NEWNET):
        try:
            unshare(flags)
        except OSError:
            continue
        return True
    return False


# --- Landlock ----------------------------------------------------------------------------------

_FS_EXECUTE, _FS_WRITE_FILE, _FS_READ_FILE, _FS_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_FS_TRUNCATE, _FS_IOCTL_DEV = 1 << 14, 1 << 15
_FS_ALL = {1: (1 << 13) - 1, 2: (1 << 14) - 1, 3: (1 << 15) - 1, 4: (1 << 15) - 1}
_FS_ALL_LATEST = (1 << 16) - 1
_NET_BIND_TCP, _NET_CONNECT_TCP = 1 << 0, 1 << 1
_SCOPE_ABSTRACT_UNIX_SOCKET, _SCOPE_SIGNAL = 1 << 0, 1 << 1
_RULE_PATH_BENEATH = 1
_READ = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
_FILE_ONLY = _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_TRUNCATE | _FS_IOCTL_DEV

SYSTEM_READ_PATHS = ("/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/etc", "/opt")
SYSTEM_READ_FILES = ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random")
# Of /proc, only the process's own directory and the machine's description: never another
# process's, such as the worker's environment.
PROC_READ_FILES = ("/proc/cpuinfo", "/proc/meminfo", "/proc/stat")


def landlock_abi() -> int:
    if _arch() is None:
        return 0
    libc = _libc()
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    return int(abi) if abi > 0 else 0


def default_read_paths() -> list[str]:
    """What a parser may read besides its document: Python, the import path, the system's
    libraries and data, models cached under ``~/.cache``, and the CPU's description (numpy)."""
    paths = [sys.prefix, sys.base_prefix, sys.exec_prefix, *sys.path, *SYSTEM_READ_PATHS]
    paths += [f"/proc/{os.getpid()}", *PROC_READ_FILES, "/sys/devices/system/cpu"]
    home = os.environ.get("HOME")
    if home:
        paths.append(str(Path(home) / ".cache"))
    return paths


def restrict_filesystem(
    read: Iterable[str], read_files: Iterable[str], write: Iterable[str]
) -> int:
    """Apply Landlock; returns the ABI applied, or 0 if the kernel has none."""
    abi = landlock_abi()
    if abi <= 0:
        return 0
    libc = _libc()
    libc.syscall.restype = ctypes.c_long
    handled_fs = _FS_ALL.get(abi, _FS_ALL_LATEST)
    attr = struct.pack("=Q", handled_fs)
    if abi >= 4:
        attr += struct.pack("=Q", _NET_BIND_TCP | _NET_CONNECT_TCP)
    if abi >= 6:
        attr += struct.pack("=Q", _SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL)
    buffer = ctypes.create_string_buffer(attr, len(attr))
    ruleset = int(libc.syscall(_LANDLOCK_CREATE_RULESET, buffer, ctypes.c_size_t(len(attr)), 0))
    if ruleset < 0:
        return 0
    try:

        def allow(path: str, access: int) -> None:
            try:
                descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                return
            try:
                if not os.path.isdir(path):
                    access &= _FILE_ONLY
                rule = struct.pack("=Qi", access & handled_fs, descriptor)
                rule_buffer = ctypes.create_string_buffer(rule, len(rule))
                libc.syscall(_LANDLOCK_ADD_RULE, ruleset, _RULE_PATH_BENEATH, rule_buffer, 0)
            finally:
                os.close(descriptor)

        for path in dict.fromkeys(p for p in read if p):
            allow(path, _READ)
        for path in dict.fromkeys(read_files):
            allow(path, _FS_READ_FILE | _FS_WRITE_FILE)
        for path in dict.fromkeys(write):
            allow(path, handled_fs)
        if not _no_new_privs(libc):
            return 0
        if libc.syscall(_LANDLOCK_RESTRICT_SELF, ruleset, 0) != 0:
            return 0
        return abi
    finally:
        os.close(ruleset)


# --- the whole of it ---------------------------------------------------------------------------


def confine(*, document: str | None, scratch: str) -> Confinement:
    """Confine this process as far as the kernel allows, then say how far that was."""
    if _arch() is None:
        return Confinement()
    namespace = isolate_network()
    read = default_read_paths()
    if document:
        read.append(document)
    landlock = restrict_filesystem(read, SYSTEM_READ_FILES, [scratch])
    seccomp = install_seccomp()
    network = "namespace" if namespace else ("seccomp" if seccomp else "none")
    return Confinement(network=network, seccomp=seccomp, landlock=landlock)


def harden_process() -> bool:
    """Make this process undumpable: a process of the same user, such as a parser, can't read
    its memory or environment through ``/proc`` or attach to it. For the worker and the API."""
    if not sys.platform.startswith("linux"):
        return False
    return _libc().prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) == 0


# --- the self-test -----------------------------------------------------------------------------


def probe(payload: dict[str, Any]) -> dict[str, bool]:
    """Try every way out a hostile parser might take; True where one worked. The self-test runs
    it in the sandbox (``stratafi_components.sandbox``)."""
    import socket

    def attempt(action: Callable[[], object]) -> bool:
        try:
            action()
        except (OSError, ValueError):
            return False
        return True

    def tcp() -> None:
        with socket.create_connection((payload["host"], int(payload["port"])), timeout=2):
            pass

    def udp() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(b"x", (payload["host"], int(payload["udp_port"])))

    def unix() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            sock.connect(payload["unix_path"])

    def secret() -> None:
        with Path(payload["secret_path"]).open("rb") as handle:
            handle.read(1)

    def environ() -> None:
        with Path(f"/proc/{os.getppid()}/environ").open("rb") as handle:
            if not handle.read(1):
                raise OSError("empty")

    def write_outside() -> None:
        with Path(payload["outside_path"]).open("wb") as handle:
            handle.write(b"x")

    return {
        "tcp": attempt(tcp),
        "udp": attempt(udp),
        "unix": attempt(unix),
        "secret": attempt(secret),
        "parent_environ": attempt(environ),
        "write_outside": attempt(write_outside),
    }


@contextlib.contextmanager
def probe_targets(directory: Path) -> Iterator[dict[str, Any]]:
    """Listeners and files for ``probe``: a TCP and a UDP socket on loopback, a Unix socket, a
    secret file and a path outside the sandbox, all in ``directory``."""
    import socket
    import threading

    tcp = socket.create_server(("127.0.0.1", 0))
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    unix_path = directory / "worker.sock"
    unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unix.bind(str(unix_path))
    unix.listen()
    stop = threading.Event()

    def accept(server: socket.socket) -> None:
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                connection, _ = server.accept()
                connection.close()
            except OSError:
                continue

    threads = [threading.Thread(target=accept, args=(s,), daemon=True) for s in (tcp, unix)]
    for thread in threads:
        thread.start()
    secret = directory / "secret.env"
    secret.write_text("STRATAFI_DATABASE_URL=postgresql://probe\n", encoding="utf-8")
    try:
        yield {
            "host": "127.0.0.1",
            "port": tcp.getsockname()[1],
            "udp_port": udp.getsockname()[1],
            "unix_path": str(unix_path),
            "secret_path": str(secret),
            "outside_path": str(directory / "planted"),
        }
    finally:
        stop.set()
        for thread in threads:
            thread.join()
        for server in (tcp, udp, unix):
            server.close()


def self_test(mode: str = "required") -> dict[str, Any]:
    """Run the probe in the sandbox here: what applied and what it reached. For a deployment to
    check its parsers are confined: ``python -m stratafi_components.confine``."""
    import tempfile

    from stratafi_components.sandbox import run_isolated

    stats: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as directory, probe_targets(Path(directory)) as targets:
        reached = run_isolated(
            "stratafi_components.confine:probe",
            targets,
            timeout=30,
            confinement=mode,  # type: ignore[arg-type]
            stats=stats,
        )
    return {"confinement": stats.get("confinement"), "reached": reached}


if __name__ == "__main__":
    import json

    try:
        report = self_test()
    except Exception as error:
        print(json.dumps({"error": getattr(error, "code", type(error).__name__)}))
        sys.exit(1)
    print(json.dumps(report))
    sys.exit(1 if any(report["reached"].values()) else 0)
